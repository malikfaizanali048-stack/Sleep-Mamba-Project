path = "src/train_kfold.py"
with open(path, "r") as f:
    content = f.read()

content = content.replace(
    "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)",
    "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)"
)
with open(path, "w") as f:
    f.write(content)
print("Patched.")
