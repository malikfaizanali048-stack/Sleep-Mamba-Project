"""
auto_save_kaggle_dataset.py
Runs ablation training, then automatically packages results/checkpoints
and creates/updates a Kaggle Dataset -- no manual steps, works even if
you're away when it finishes.
"""

import subprocess
import os
import json
from kaggle_secrets import UserSecretsClient

# --- Step 1: run training ---
subprocess.run(["python", "src/train_ablation.py"])

print("\n" + "="*60)
print("Training step finished. Auto-saving to Kaggle Dataset...")
print("="*60)

# --- Step 2: set up Kaggle API credentials from secrets ---
secrets = UserSecretsClient()
kaggle_username = secrets.get_secret("malikfaizan029")
kaggle_key = secrets.get_secret("9762c3b23f4db1b07ad3775eb3abd207")

os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
    json.dump({"username": kaggle_username, "key": kaggle_key}, f)
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

# --- Step 3: prepare a folder with everything worth saving ---
SAVE_DIR = "/kaggle/working/auto_save_bundle"
os.makedirs(SAVE_DIR, exist_ok=True)
subprocess.run(["cp", "-r", "results", SAVE_DIR])
subprocess.run(["cp", "-r", "checkpoints", SAVE_DIR])

# --- Step 4: dataset metadata (required by Kaggle API) ---
DATASET_SLUG = "sleepmamba-training-results"
metadata = {
    "title": "SleepMamba Training Results (auto-saved)",
    "id": f"{kaggle_username}/{DATASET_SLUG}",
    "licenses": [{"name": "CC0-1.0"}]
}
with open(os.path.join(SAVE_DIR, "dataset-metadata.json"), "w") as f:
    json.dump(metadata, f)

# --- Step 5: create dataset if it doesn't exist, else create new version ---
check = subprocess.run(
    ["kaggle", "datasets", "status", f"{kaggle_username}/{DATASET_SLUG}"],
    capture_output=True, text=True
)

if check.returncode != 0:
    print("Dataset doesn't exist yet -- creating it for the first time...")
    result = subprocess.run(
        ["kaggle", "datasets", "create", "-p", SAVE_DIR, "-r", "zip"],
        capture_output=True, text=True
    )
else:
    print("Dataset exists -- pushing new version...")
    result = subprocess.run(
        ["kaggle", "datasets", "version", "-p", SAVE_DIR, "-m",
         "Auto-save after training run", "-r", "zip"],
        capture_output=True, text=True
    )

print(result.stdout)
print(result.stderr)

if result.returncode == 0:
    print("\n✓ Successfully saved to Kaggle Dataset. Safe to close notebook.")
else:
    print("\n✗ Kaggle Dataset save failed -- check errors above!")
