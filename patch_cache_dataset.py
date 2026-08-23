path = "src/train_ablation.py"
with open(path, "r") as f:
    content = f.read()

old_sig = "def run_single_experiment(ablation, run_idx, train_files, test_files, device, seed):"
new_sig = "def run_single_experiment(ablation, run_idx, train_ds, test_ds, device, seed):"
content = content.replace(old_sig, new_sig)

old_load = '''    print(f"    Loading train dataset ({len(train_files)} files)...", flush=True)
    train_ds = SleepWindowDataset(PROCESSED_DIR, train_files, T)
    print(f"    Loading test dataset ({len(test_files)} files)...", flush=True)
    test_ds = SleepWindowDataset(PROCESSED_DIR, test_files, T)
    print(f"    Loaded: {len(train_ds)} train windows, {len(test_ds)} test windows", flush=True)

'''
content = content.replace(old_load, "")

old_call = '''            r = run_single_experiment(ablation, run_idx, train_files, test_files,
                                       device, seed=42 + run_idx)'''
new_call = '''            r = run_single_experiment(ablation, run_idx, train_ds, test_ds,
                                       device, seed=42 + run_idx)'''
content = content.replace(old_call, new_call)

old_main = '''    train_files = splits[0]["train_files"]
    test_files = splits[0]["test_files"]'''
new_main = '''    train_files = splits[0]["train_files"]
    test_files = splits[0]["test_files"]

    print("Loading dataset once (reused across all ablation runs)...", flush=True)
    train_ds = SleepWindowDataset(PROCESSED_DIR, train_files, T)
    test_ds = SleepWindowDataset(PROCESSED_DIR, test_files, T)
    print(f"Loaded: {len(train_ds)} train windows, {len(test_ds)} test windows", flush=True)'''
content = content.replace(old_main, new_main)

with open(path, "w") as f:
    f.write(content)
print("Patched.")
