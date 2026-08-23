path = "src/train_ablation.py"
with open(path, "r") as f:
    content = f.read()

old = '''    train_ds = SleepWindowDataset(PROCESSED_DIR, train_files, T)
    test_ds = SleepWindowDataset(PROCESSED_DIR, test_files, T)'''

new = '''    print(f"    Loading train dataset ({len(train_files)} files)...", flush=True)
    train_ds = SleepWindowDataset(PROCESSED_DIR, train_files, T)
    print(f"    Loading test dataset ({len(test_files)} files)...", flush=True)
    test_ds = SleepWindowDataset(PROCESSED_DIR, test_files, T)
    print(f"    Loaded: {len(train_ds)} train windows, {len(test_ds)} test windows", flush=True)'''

content = content.replace(old, new)
with open(path, "w") as f:
    f.write(content)
print("Patched.")
