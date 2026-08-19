"""
dam.py
Dual-Axis Mamba (DAM) Block.
"""

import torch
import torch.nn as nn

from selective_ssm import MambaBlock


class DualAxisMambaBlock(nn.Module):
    def __init__(self, latent_dim: int, temporal_len: int, d_state: int = 16):
        super().__init__()
        self.D = latent_dim
        self.E = temporal_len

        self.gate_proj = nn.Linear(self.E, self.E)
        self.ssm_temporal = MambaBlock(d_model=self.D, d_state=d_state)
        self.ssm_modal = MambaBlock(d_model=self.E, d_state=d_state)

    def forward_no_pool(self, F_double_prime):
        B, D, E = F_double_prime.shape
        assert D == self.D and E == self.E

        V_i = torch.sigmoid(self.gate_proj(F_double_prime))

        temp_in = F_double_prime.transpose(1, 2)
        temp_out = self.ssm_temporal(temp_in)
        temp_out = temp_out.transpose(1, 2)
        Z_intra = temp_out * V_i

        mod_in = F_double_prime
        mod_out = self.ssm_modal(mod_in)
        Z_inter = mod_out * V_i

        Z_i = Z_intra + Z_inter
        return Z_i

    def forward(self, F_double_prime):
        Z_i = self.forward_no_pool(F_double_prime)
        g_i = Z_i.mean(dim=2)
        return g_i


if __name__ == "__main__":
    B, D, E = 4, 128, 15
    dummy_input = torch.randn(B, D, E)
    dam = DualAxisMambaBlock(latent_dim=D, temporal_len=E, d_state=16)
    g_i = dam(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape (epoch descriptor g_i): {g_i.shape}")
    print(f"Num parameters: {sum(p.numel() for p in dam.parameters()):,}")