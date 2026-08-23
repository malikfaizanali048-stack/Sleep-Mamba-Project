"""
auto_save_and_train.py
Runs the ablation training, then automatically commits + pushes all
results/checkpoints to GitHub when done -- no manual action needed,
safe even if you're away when it finishes.
"""

import subprocess
import os

# Run the actual training
result = subprocess.run(["python", "src/train_ablation.py"])

# Regardless of whether it fully completed or was interrupted partway,
# save whatever progress exists (results/ablation/*.json are saved
# incrementally per-variant, so partial progress is still valuable)
print("\n" + "="*60)
print("Training step finished (or was interrupted). Auto-saving to GitHub...")
print("="*60)

from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")

subprocess.run(["git", "add", "."])
subprocess.run(["git", "commit", "-m", "Auto-save: ablation training progress"])
push_url = f"https://malikfaizanali048-stack:{token}@github.com/malikfaizanali048-stack/Sleep-Mamba-Project.git"
push_result = subprocess.run(["git", "push", push_url])

if push_result.returncode == 0:
    print("\n✓ Successfully pushed to GitHub. Safe to close notebook.")
else:
    print("\n✗ Push failed -- results may only exist in this session!")
