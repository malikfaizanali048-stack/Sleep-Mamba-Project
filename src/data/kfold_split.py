"""
kfold_split.py
Subject-level k-fold CV splitting (k=10 for SleepEDF-78, Section 4.2).
"""

import os
import numpy as np
from sklearn.model_selection import KFold

PROCESSED_DIR = "data/processed"
K_FOLDS = 10
SEED = 42


def get_subject_ids(processed_dir):
    files = [f for f in os.listdir(processed_dir) if f.endswith(".npz")]
    subject_ids = sorted(set(f[:5] for f in files))
    return subject_ids


def get_recordings_for_subject(processed_dir, subject_id):
    files = [f for f in os.listdir(processed_dir)
             if f.startswith(subject_id) and f.endswith(".npz")]
    return sorted(files)


def make_kfold_splits(processed_dir, k=K_FOLDS, seed=SEED):
    subject_ids = get_subject_ids(processed_dir)
    print(f"Found {len(subject_ids)} unique subjects.")

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    subject_ids = np.array(subject_ids)

    splits = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(subject_ids)):
        train_subjects = subject_ids[train_idx].tolist()
        test_subjects = subject_ids[test_idx].tolist()

        train_files = []
        for sid in train_subjects:
            train_files.extend(get_recordings_for_subject(processed_dir, sid))

        test_files = []
        for sid in test_subjects:
            test_files.extend(get_recordings_for_subject(processed_dir, sid))

        splits.append({
            "fold": fold_idx,
            "train_subjects": train_subjects,
            "test_subjects": test_subjects,
            "train_files": train_files,
            "test_files": test_files,
        })

        print(f"Fold {fold_idx}: {len(train_subjects)} train subjects "
              f"({len(train_files)} files), {len(test_subjects)} test subjects "
              f"({len(test_files)} files)")

    return splits


if __name__ == "__main__":
    splits = make_kfold_splits(PROCESSED_DIR, k=K_FOLDS, seed=SEED)
    print(f"\nGenerated {len(splits)} folds successfully.")