"""
selective_ssm.py
Dual-compatible Mamba Block:
- Uses official CUDA-accelerated `mamba_ssm` on Linux/Kaggle/GPU.
- Uses a lightweight linear stub on Windows/CPU for local testing.
"""
import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba  # type: ignore
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    # Lightweight CPU fallback for local debugging on Windows
    class Mamba(nn.Module):
        def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
            super().__init__()
            self.proj = nn.Linear(d_model, d_model)

        def forward(self, x):
            return self.proj(x)


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_inner: int = None, d_state: int = 16, conv_kernel: int = 4): # type: ignore
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=conv_kernel,
            expand=2,
        )

    def forward(self, x):
        return self.mamba(x)


if __name__ == "__main__":
    print(f"Mamba CUDA acceleration available: {MAMBA_AVAILABLE}")
    x = torch.randn(2, 15, 128)
    block = MambaBlock(d_model=128, d_state=16)
    y = block(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print("Local CPU test successful!")