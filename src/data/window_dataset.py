"""
windowed_dataset.py
PyTorch Dataset building non-overlapping T-epoch windows from subject .npz files.
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
            path = os.path.join(processed_dir, fname)
            data = np.load(path)
            signals, labels = data["signals"], data["labels"]

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