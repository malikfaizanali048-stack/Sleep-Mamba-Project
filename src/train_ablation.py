"""
train_ablation.py
Component ablation study (paper Table 7): 3 independent runs per variant,
fixed T=5, single fixed train/test split (fold 0's split). Dataset loaded
ONCE and reused across all 15 runs (5 variants x 3 runs).
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "models"))
sys.path.append(current_dir)

from models.sleepmamba_ablation import SleepMambaAblation
from data.kfold_split import make_kfold_splits
from data.windowed_dataset import SleepWindowDataset

PROCESSED_DIR = "/kaggle/input/datasets/malikfaizan029/sleep-edf-processed-normalized"
RESULTS_DIR = "results/ablation"
T = 5
BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 100
PATIENCE = 10
N_RUNS = 3

ABLATIONS = ["none", "no_sdam", "no_dam_intra", "no_dam_inter", "no_sbm"]


def compute_metrics(preds, labels):
    if len(preds) == 0:
        return 0.0, 0.0, 0.0
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return (accuracy_score(labels, preds), cohen_kappa_score(labels, preds),
            f1_score(labels, preds, average="macro"))


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss, total_count = 0.0, 0
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
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * X.size(0)
            total_count += X.size(0)
            all_preds.append(logits.argmax(dim=-1).detach().cpu().numpy().reshape(-1))
            all_labels.append(Y.detach().cpu().numpy().reshape(-1))

    avg_loss = total_loss / max(total_count, 1)
    acc, kappa, mf1 = compute_metrics(all_preds, all_labels)
    return avg_loss, acc, kappa, mf1


def run_single_experiment(ablation, run_idx, train_ds, test_ds, device, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=2, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True, drop_last=False)

    model = SleepMambaAblation(in_channels=3, latent_dim=128, temporal_len=15,
                                d_state=16, num_classes=5, ablation=ablation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
                                    betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.CrossEntropyLoss()

    best_mf1 = -1.0
    best_state = None
    patience_counter = 0
    epoch = 0

    for epoch in range(MAX_EPOCHS):
        train_loss, train_acc, train_kappa, train_mf1 = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True)
        test_loss, test_acc, test_kappa, test_mf1 = run_epoch(
            model, test_loader, optimizer, criterion, device, train=False)

        if test_mf1 > best_mf1:
            best_mf1 = test_mf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"    [{ablation}] run {run_idx} epoch {epoch+1}: "
              f"train_loss={train_loss:.4f} test_mf1={test_mf1*100:.2f}% "
              f"(best={best_mf1*100:.2f}%, patience={patience_counter}/{PATIENCE})",
              flush=True)

        if patience_counter >= PATIENCE:
            break

    model.load_state_dict(best_state)
    _, final_acc, final_kappa, final_mf1 = run_epoch(
        model, test_loader, optimizer, criterion, device, train=False)

    print(f"  [{ablation}] run {run_idx}: Acc={final_acc*100:.2f}% "
          f"Kappa={final_kappa:.3f} MF1={final_mf1*100:.2f}% (stopped epoch {epoch+1})",
          flush=True)
    return {"acc": final_acc, "kappa": final_kappa, "mf1": final_mf1}


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    splits = make_kfold_splits(PROCESSED_DIR, k=10, seed=42)
    train_files = splits[0]["train_files"]
    test_files = splits[0]["test_files"]

    print("Loading dataset once (reused across all ablation runs)...", flush=True)
    train_ds = SleepWindowDataset(PROCESSED_DIR, train_files, T)
    test_ds = SleepWindowDataset(PROCESSED_DIR, test_files, T)
    print(f"Loaded: {len(train_ds)} train windows, {len(test_ds)} test windows", flush=True)

    all_results = {}
    for ablation in ABLATIONS:
        result_path = os.path.join(RESULTS_DIR, f"{ablation}.json")
        if os.path.exists(result_path):
            print(f"{ablation}: already completed, skipping.", flush=True)
            with open(result_path) as f:
                all_results[ablation] = json.load(f)
            continue

        print(f"\n{'='*60}\nAblation: {ablation}\n{'='*60}", flush=True)
        runs = []
        for run_idx in range(N_RUNS):
            r = run_single_experiment(ablation, run_idx, train_ds, test_ds,
                                       device, seed=42 + run_idx)
            runs.append(r)

        accs = [r["acc"] for r in runs]
        kappas = [r["kappa"] for r in runs]
        mf1s = [r["mf1"] for r in runs]

        summary = {
            "ablation": ablation,
            "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "kappa_mean": float(np.mean(kappas)), "kappa_std": float(np.std(kappas)),
            "mf1_mean": float(np.mean(mf1s)), "mf1_std": float(np.std(mf1s)),
        }
        with open(result_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"{ablation}: Acc={summary['acc_mean']*100:.2f}%+/-{summary['acc_std']*100:.2f}% "
              f"Kappa={summary['kappa_mean']:.3f}+/-{summary['kappa_std']:.3f} "
              f"MF1={summary['mf1_mean']*100:.2f}%+/-{summary['mf1_std']*100:.2f}%", flush=True)
        all_results[ablation] = summary

    print(f"\n{'='*60}\nALL ABLATION RESULTS\n{'='*60}")
    for ablation, r in all_results.items():
        print(f"{ablation:15s}: Acc={r['acc_mean']*100:.2f}% "
              f"Kappa={r['kappa_mean']:.3f} MF1={r['mf1_mean']*100:.2f}%")
