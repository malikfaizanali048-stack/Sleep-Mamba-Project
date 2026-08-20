"""
windowed_dataset.py
PyTorch Dataset building non-overlapping T-epoch windows from subject .npz files.
Handles both possible key naming conventions ("signals"/"labels" or "x"/"y")
so it works regardless of which preprocessing run produced the data.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SleepWindowDataset(Dataset):
    def __init__(self, processed_dir, file_list, T):
        self.T = T
        self.windows = []

        for fname in file_list:
            path = os.path.join(processed_dir, fname) if not fname.startswith("/") else fname
            if not os.path.exists(path):
                path = os.path.join(processed_dir, os.path.basename(fname))
            if not os.path.exists(path):
                continue

            data = np.load(path)
            keys = data.files

            if "signals" in keys:
                signals = data["signals"]
            elif "x" in keys:
                signals = data["x"]
            else:
                raise KeyError(f"No recognized signal key in {fname}. Found keys: {keys}")

            if "labels" in keys:
                labels = data["labels"]
            elif "y" in keys:
                labels = data["y"]
            else:
                raise KeyError(f"No recognized label key in {fname}. Found keys: {keys}")

            n_epochs = signals.shape[0]
            n_windows = n_epochs // T
            if n_windows == 0:
                continue

            sig_w = signals[: n_windows * T].reshape(n_windows, T, *signals.shape[1:])
            lab_w = labels[: n_windows * T].reshape(n_windows, T)

            for i in range(n_windows):
                self.windows.append((sig_w[i], lab_w[i]))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sig, lab = self.windows[idx]
        return torch.from_numpy(sig).float(), torch.from_numpy(lab).long()
