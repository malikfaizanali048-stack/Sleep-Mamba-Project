"""
preprocess_batch.py
Runs preprocess.py logic over ALL currently-available subjects, skipping
already-processed ones. Includes normalization + missing-file resilience.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import (
    load_subject, annotations_to_epoch_labels, trim_wake_padding,
    segment_signal, find_subject_pairs, RAW_DIR, OUT_DIR
)


def process_subject(psg_path, hyp_path, subject_id, out_dir):
    out_path = os.path.join(out_dir, f"{subject_id}.npz")
    if os.path.exists(out_path):
        print(f"SKIP (already processed): {subject_id}")
        return

    print(f"Processing {subject_id}...")
    raw, annotations = load_subject(psg_path, hyp_path)
    total_duration_sec = raw.times[-1]

    epoch_labels_full = annotations_to_epoch_labels(annotations, total_duration_sec)
    epoch_labels_trimmed, (start, end) = trim_wake_padding(epoch_labels_full)

    if len(epoch_labels_trimmed) == 0:
        print(f"  WARNING: no usable epochs for {subject_id}, skipping.")
        return

    keep_mask = epoch_labels_trimmed != -1
    valid_positions = np.where(keep_mask)[0]

    signals = segment_signal(raw, start, end)
    signals = signals[valid_positions]
    labels = epoch_labels_trimmed[valid_positions]

    mean = signals.mean(axis=(0, 2), keepdims=True)
    std = signals.std(axis=(0, 2), keepdims=True) + 1e-8
    signals = ((signals - mean) / std).astype(np.float32)

    np.savez_compressed(out_path, signals=signals, labels=labels)
    print(f"  Saved {out_path} | shape={signals.shape} | "
          f"label counts={np.bincount(labels, minlength=5)}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    pairs = find_subject_pairs(RAW_DIR)
    print(f"Found {len(pairs)} subject pair(s) currently available.\n")

    for psg_path, hyp_path, subject_id in pairs:
        try:
            process_subject(psg_path, hyp_path, subject_id, OUT_DIR)
        except Exception as e:
            print(f"  ERROR processing {subject_id}: {e}")

    print("\nDone.")