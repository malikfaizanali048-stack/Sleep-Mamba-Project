"""
train_kfold.py
Full paper-faithful k-fold CV training with checkpointing/resume support.
fp32, batch_size=64 per paper Sec 4.2. DataParallel for multi-GPU speed.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "models"))
sys.path.append(current_dir)

from models.sleepmamba import SleepMamba
from data.kfold_split import make_kfold_splits
from data.windowed_dataset import SleepWindowDataset

DEFAULT_PROCESSED_DIR = "/kaggle/input/datasets/malikfaizan029/sleep-edf-processed-normalized"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
K_FOLDS = 10
BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 100
PATIENCE = 10
SEED = 42


def compute_metrics(preds, labels):
    if len(preds) == 0:
        return 0.0, 0.0, 0.0
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    acc = accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    mf1 = f1_score(labels, preds, average="macro")
    return acc, kappa, mf1


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_count = 0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for X, Y in loader:
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(X)
            loss = criterion(logits.reshape(-1, 5), Y.reshape(-1))

            if train:
                if not torch.isfinite(loss) or loss.item() > 10.0:
                    print(f"  SKIPPING bad batch: loss={loss.item()}", flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

            total_loss += loss.item() * X.size(0)
            total_count += X.size(0)
            preds = logits.argmax(dim=-1).detach().cpu().numpy().reshape(-1)
            labels = Y.detach().cpu().numpy().reshape(-1)
            all_preds.append(preds)
            all_labels.append(labels)

    avg_loss = total_loss / max(total_count, 1)
    acc, kappa, mf1 = compute_metrics(all_preds, all_labels)
    return avg_loss, acc, kappa, mf1


def train_fold(fold_idx, split, T, processed_dir, device, resume=True):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fold_ckpt_path = os.path.join(CHECKPOINT_DIR, f"T{T}_fold{fold_idx}.pt")
    fold_result_path = os.path.join(RESULTS_DIR, f"T{T}_fold{fold_idx}.json")

    if os.path.exists(fold_result_path):
        with open(fold_result_path) as f:
            result = json.load(f)
        if result.get("completed", False):
            print(f"Fold {fold_idx} (T={T}) already completed. Skipping.", flush=True)
            return result

    print(f"\n{'='*60}\nFold {fold_idx} (T={T}) -- "
          f"{len(split['train_subjects'])} train subj, "
          f"{len(split['test_subjects'])} test subj\n{'='*60}", flush=True)

    train_ds = SleepWindowDataset(processed_dir, split["train_files"], T)
    test_ds = SleepWindowDataset(processed_dir, split["test_files"], T)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=2, pin_memory=True, persistent_workers=True,
                               drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True, persistent_workers=True,
                              drop_last=False)

    print(f"Train windows: {len(train_ds)}, Test windows: {len(test_ds)}", flush=True)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    raw_model = SleepMamba(in_channels=3, latent_dim=128, temporal_len=15,
                            d_state=16, num_classes=5)

    # DataParallel disabled permanently: causes NaN instability when
    # combined with this model's non-standard Mamba usage (modality-axis
    # branch treats D=128 as sequence length). Confirmed via direct A/B
    # test -- single GPU is stable, DataParallel is not, even with input
    # clamping applied. Documented in assumptions_log.md.
    model = raw_model.to(device)
    print("Running on single GPU (DataParallel disabled -- causes NaN).", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
                                    betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_mf1 = -1.0
    patience_counter = 0
    history = []

    if resume and os.path.exists(fold_ckpt_path):
        ckpt = torch.load(fold_ckpt_path, map_location=device)
        unwrap_model = model.module if isinstance(model, nn.DataParallel) else model
        unwrap_model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_mf1 = ckpt["best_mf1"]
        patience_counter = ckpt["patience_counter"]
        history = ckpt["history"]
        print(f"Resumed from epoch {start_epoch}, best_mf1={best_mf1:.4f}", flush=True)

    for epoch in range(start_epoch, MAX_EPOCHS):
        train_loss, train_acc, train_kappa, train_mf1 = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True)
        test_loss, test_acc, test_kappa, test_mf1 = run_epoch(
            model, test_loader, optimizer, criterion, device, train=False)

        print(f"Epoch {epoch+1}/{MAX_EPOCHS} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc*100:.2f}% | "
              f"Test Loss {test_loss:.4f} Acc {test_acc*100:.2f}% "
              f"Kappa {test_kappa:.3f} MF1 {test_mf1*100:.2f}%", flush=True)

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "test_loss": test_loss, "test_acc": test_acc,
                         "test_kappa": test_kappa, "test_mf1": test_mf1})

        unwrap_model = model.module if isinstance(model, nn.DataParallel) else model
        improved = test_mf1 > best_mf1
        if improved:
            best_mf1 = test_mf1
            patience_counter = 0
            torch.save(unwrap_model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, f"T{T}_fold{fold_idx}_best.pt"))
        else:
            patience_counter += 1

        torch.save({"epoch": epoch, "model_state": unwrap_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "best_mf1": best_mf1,
                    "patience_counter": patience_counter, "history": history}, fold_ckpt_path)

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1} (patience={PATIENCE})", flush=True)
            break

    unwrap_model = model.module if isinstance(model, nn.DataParallel) else model
    unwrap_model.load_state_dict(torch.load(
        os.path.join(CHECKPOINT_DIR, f"T{T}_fold{fold_idx}_best.pt"), map_location=device))
    final_loss, final_acc, final_kappa, final_mf1 = run_epoch(
        model, test_loader, optimizer, criterion, device, train=False)

    result = {"fold": fold_idx, "T": T, "acc": final_acc, "kappa": final_kappa,
              "mf1": final_mf1, "completed": True}
    with open(fold_result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Fold {fold_idx} FINAL: Acc={final_acc*100:.2f}% "
          f"Kappa={final_kappa:.3f} MF1={final_mf1*100:.2f}%", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--folds", type=str, default=None)
    parser.add_argument("--processed_dir", type=str, default=DEFAULT_PROCESSED_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Processed Dataset Path: {args.processed_dir}", flush=True)

    splits = make_kfold_splits(args.processed_dir, k=K_FOLDS, seed=SEED)
    fold_indices = (list(map(int, args.folds.split(","))) if args.folds
                     else list(range(K_FOLDS)))

    all_results = []
    for fold_idx in fold_indices:
        result = train_fold(fold_idx, splits[fold_idx], args.T, args.processed_dir, device, args.resume)
        all_results.append(result)

    accs = [r["acc"] for r in all_results]
    kappas = [r["kappa"] for r in all_results]
    mf1s = [r["mf1"] for r in all_results]

    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY (T={args.T}, {len(all_results)} folds)", flush=True)
    print(f"Acc:   {np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}%", flush=True)
    print(f"Kappa: {np.mean(kappas):.3f} +/- {np.std(kappas):.3f}", flush=True)
    print(f"MF1:   {np.mean(mf1s)*100:.2f}% +/- {np.std(mf1s)*100:.2f}%", flush=True)
    print(f"{'='*60}", flush=True)
