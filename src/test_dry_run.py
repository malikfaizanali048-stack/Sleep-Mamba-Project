"""
test_dry_run.py
Runs a 1-epoch, 2-fold dry run using synthetic mock dataset 
to verify loss calculation, optimizer steps, and evaluation metrics.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.sleepmamba import SleepMamba

def run_dry_run():
    print("--- Starting SleepMamba Local Dry-Run ---")
    
    # 1. Generate Synthetic Data
    # Batch size = 4, Sequence Length = 5 epochs, Channels = 3, Sample Length = 3000
    num_samples = 20
    B, T, C, L = 4, 5, 3, 3000
    num_classes = 5

    mock_x = torch.randn(num_samples, T, C, L)
    mock_y = torch.randint(0, num_classes, (num_samples, T))

    dataset = TensorDataset(mock_x, mock_y)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 2. Instantiate Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SleepMamba(in_channels=C, latent_dim=128, temporal_len=15, d_state=16, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 3. Simulate Training Epoch
    model.train()
    total_loss = 0.0
    print("\n[Training Loop Simulation]")
    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass: logits shape [B, T, C_out]
        logits = model(x_batch) 
        
        # Reshape for CrossEntropyLoss: [B*T, num_classes] vs [B*T]
        loss = criterion(logits.view(-1, num_classes), y_batch.view(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"  Batch {batch_idx + 1}/{len(dataloader)} | Loss: {loss.item():.4f}")

    # 4. Simulate Evaluation Epoch & Metric Calculation
    model.eval()
    all_preds, all_targets = [], []
    print("\n[Validation Loop Simulation]")
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.append(preds.cpu())
            all_targets.append(y_batch)

    all_preds = torch.cat(all_preds, dim=0).view(-1)
    all_targets = torch.cat(all_targets, dim=0).view(-1)

    acc = (all_preds == all_targets).float().mean().item()
    print(f"\nDry-run completed successfully!")
    print(f"  Avg Training Loss: {total_loss / len(dataloader):.4f}")
    print(f"  Mock Accuracy:     {acc * 100:.2f}%")

if __name__ == "__main__":
    run_dry_run()