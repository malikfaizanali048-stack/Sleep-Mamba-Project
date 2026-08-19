"""
sequence_bi_mamba.py
Sequence Bi-Mamba (SBM) Layer.
"""

import torch
import torch.nn as nn

from selective_ssm import MambaBlock


class SequenceBiMamba(nn.Module):
    def __init__(self, latent_dim: int, d_state: int = 16):
        super().__init__()
        self.D = latent_dim
        self.mamba_fwd = MambaBlock(d_model=self.D, d_state=d_state)
        self.mamba_bwd = MambaBlock(d_model=self.D, d_state=d_state)

    def forward(self, G):
        O_fwd = self.mamba_fwd(G)
        G_flipped = torch.flip(G, dims=[1])
        O_bwd_flipped = self.mamba_bwd(G_flipped)
        O_bwd = torch.flip(O_bwd_flipped, dims=[1])
        O = O_fwd + O_bwd
        return O


if __name__ == "__main__":
    B, T, D = 2, 5, 128
    dummy_input = torch.randn(B, T, D)
    sbm = SequenceBiMamba(latent_dim=D, d_state=16)
    O = sbm(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {O.shape}")
    print(f"Num parameters: {sum(p.numel() for p in sbm.parameters()):,}")