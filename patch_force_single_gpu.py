path = "src/train_kfold.py"
with open(path, "r") as f:
    content = f.read()

old = '''    if torch.cuda.device_count() > 1:
        print(f"Enabling DataParallel across {torch.cuda.device_count()} GPUs!", flush=True)
        model = nn.DataParallel(raw_model).to(device)
    else:
        model = raw_model.to(device)'''

new = '''    # DataParallel disabled permanently: causes NaN instability when
    # combined with this model's non-standard Mamba usage (modality-axis
    # branch treats D=128 as sequence length). Confirmed via direct A/B
    # test -- single GPU is stable, DataParallel is not, even with input
    # clamping applied. Documented in assumptions_log.md.
    model = raw_model.to(device)
    print("Running on single GPU (DataParallel disabled -- causes NaN).", flush=True)'''

content = content.replace(old, new)
with open(path, "w") as f:
    f.write(content)
print("Patched.")
