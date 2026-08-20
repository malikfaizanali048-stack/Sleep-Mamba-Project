import torch
from mamba_ssm import Mamba

torch.manual_seed(42)
device = torch.device("cuda")

# Simulate our DAM's unusual usage: D=128 as "sequence", E=15 as "features"
model = Mamba(d_model=15, d_state=16, d_conv=4, expand=2).to(device)

for trial in range(20):
    x = torch.randn(64, 128, 15).to(device) * 2  # roughly matches clamped input scale
    x = torch.clamp(x, -10, 10)
    with torch.no_grad():
        y = model(x)
    has_nan = torch.isnan(y).any().item()
    max_val = y.abs().max().item()
    print(f"Trial {trial}: NaN={has_nan}, max_abs_output={max_val:.4f}")
    if has_nan:
        print("!!! Mamba itself produces NaN on this exact usage pattern (D=128 as seq_len)")
        break

print("Isolated Mamba test complete.")
