"""
mle.py
Multimodal Local Encoder: Multi-Scale CNN + Sparse Dual Attention Module.
"""

import torch
import torch.nn as nn


class MultiScaleBranch(nn.Module):
    def __init__(self, in_channels: int, first_kernel: int, first_stride: int,
                 first_pool: int, dropout: float = 0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=first_kernel,
                      stride=first_stride, padding=first_kernel // 2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=first_pool, stride=first_pool),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=7, stride=1, padding=3),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.5):
        super().__init__()
        self.small_branch = MultiScaleBranch(
            in_channels, first_kernel=50, first_stride=6, first_pool=4, dropout=dropout
        )
        self.large_branch = MultiScaleBranch(
            in_channels, first_kernel=400, first_stride=50, first_pool=2, dropout=dropout
        )

    def forward(self, x):
        small_out = self.small_branch(x)
        large_out = self.large_branch(x)
        target_len = min(small_out.shape[-1], large_out.shape[-1])
        small_out = nn.functional.adaptive_avg_pool1d(small_out, target_len)
        large_out = nn.functional.adaptive_avg_pool1d(large_out, target_len)
        fused = torch.cat([small_out, large_out], dim=1)
        return fused


class SparseDualAttentionModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
        )
        self.temporal_conv = nn.Conv1d(1, 1, kernel_size=1)

    def forward(self, F):
        B, D, E = F.shape
        gap = F.mean(dim=2)
        gmp = F.amax(dim=2)
        channel_logits = self.channel_mlp(gap) + self.channel_mlp(gmp)
        M_c = torch.sigmoid(channel_logits).unsqueeze(-1)
        F_prime = M_c * F
        channel_avg = F_prime.mean(dim=1, keepdim=True)
        M_t = torch.sigmoid(self.temporal_conv(channel_avg))
        F_double_prime = M_t * F_prime
        return F_double_prime


class MultimodalLocalEncoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        self.mscnn = MultiScaleCNN(in_channels, dropout=dropout)
        self.project = nn.Sequential(
            nn.Conv1d(256, latent_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.sdam = SparseDualAttentionModule(channels=latent_dim)

    def forward(self, x):
        fused = self.mscnn(x)
        F = self.project(fused)
        F_double_prime = self.sdam(F)
        return F_double_prime


if __name__ == "__main__":
    dummy_input = torch.randn(4, 3, 3000)
    model = MultimodalLocalEncoder(in_channels=3, latent_dim=128)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Num parameters: {sum(p.numel() for p in model.parameters()):,}")