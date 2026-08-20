path = "src/train_kfold.py"
with open(path, "r") as f:
    content = f.read()

old = '''    raw_model = SleepMamba(in_channels=3, latent_dim=128, temporal_len=15,
                            d_state=16, num_classes=5)'''

new = '''    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    raw_model = SleepMamba(in_channels=3, latent_dim=128, temporal_len=15,
                            d_state=16, num_classes=5)'''

content = content.replace(old, new)
with open(path, "w") as f:
    f.write(content)
print("Patched.")
