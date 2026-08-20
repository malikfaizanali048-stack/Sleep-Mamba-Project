import os
import numpy as np

DATA_DIR = "/kaggle/input/datasets/malikfaizan029/sleep-edf-processed-normalized"
bad_count = 0
total = 0

for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith(".npz"):
        continue
    total += 1
    path = os.path.join(DATA_DIR, fname)
    data = np.load(path)
    x = data["x"] if "x" in data.files else data["signals"]

    has_nan = np.isnan(x).any()
    has_inf = np.isinf(x).any()
    max_abs = np.abs(x).max()
    shape = x.shape

    if has_nan or has_inf or max_abs > 50 or shape[1] != 3:
        bad_count += 1
        print(f"BAD: {fname} | shape={shape} NaN={has_nan} Inf={has_inf} max={max_abs:.2f}")

print(f"\n{bad_count}/{total} files are problematic")
