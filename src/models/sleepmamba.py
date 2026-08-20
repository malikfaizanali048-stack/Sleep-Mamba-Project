"""
sleepmamba.py
Full SleepMamba model: MLE -> DAM x2 -> SBM x2 -> classifier.
Includes input clamping (root-cause NaN fix): rare artifact spikes in
normalized PSG data destabilize the SSM's exponential discretization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from mle import MultimodalLocalEncoder
from dam import DualAxisMambaBlock
from sequence_bi_mamba import SequenceBiMamba


class SleepMamba(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        temporal_len: int = 15,
        d_state: int = 16,
        num_classes: int = 5,
        n_dam_layers: int = 2,
        n_sbm_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.D = latent_dim
        self.E = temporal_len

        self.mle = MultimodalLocalEncoder(
            in_channels=in_channels, latent_dim=latent_dim, dropout=dropout
        )
        self.dam_layers = nn.ModuleList([
            DualAxisMambaBlock(latent_dim=latent_dim, temporal_len=temporal_len, d_state=d_state)
            for _ in range(n_dam_layers)
        ])
        # LayerNorm between stacked DAM/SBM stages: prevents magnitude
        # drift compounding across 4 stacked multiplicative stages
        # (2 DAM x 2 branches, 2 SBM x 2 branches). Not explicit in
        # paper's Fig 1 diagram; standard practice in Mamba-stack
        # architectures, added after diagnosing NaN root cause via
        # isolated-layer testing. Documented in assumptions_log.md.
        self.dam_norms = nn.ModuleList([nn.LayerNorm(temporal_len) for _ in range(n_dam_layers - 1)])
        self.post_dam_norm = nn.LayerNorm(latent_dim)

        self.sbm_layers = nn.ModuleList([
            SequenceBiMamba(latent_dim=latent_dim, d_state=d_state)
            for _ in range(n_sbm_layers)
        ])
        self.sbm_norms = nn.ModuleList([nn.LayerNorm(latent_dim) for _ in range(n_sbm_layers)])
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, X):
        # X: [B, T, C, L]
        # Clamp extreme normalization outliers (root-cause NaN fix --
        # documented in assumptions_log.md item 10)
        X = torch.clamp(X, min=-10.0, max=10.0)
        B, T, C, L = X.shape

        X_flat = X.view(B * T, C, L)
        F_pp = self.mle(X_flat)

        Z = F_pp
        for i, dam in enumerate(self.dam_layers):
            if i < len(self.dam_layers) - 1:
                Z = dam.forward_no_pool(Z)
                Z = self.dam_norms[i](Z)
            else:
                Z = dam(Z)

        g = self.post_dam_norm(Z)
        G = g.view(B, T, self.D)

        O = G
        for i, sbm in enumerate(self.sbm_layers):
            O = sbm(O)
            O = self.sbm_norms[i](O)

        logits = self.classifier(O)
        return logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, C, L = 2, 5, 3, 3000
    dummy_input = torch.randn(B, T, C, L).to(device)

    model = SleepMamba(in_channels=C, latent_dim=128, temporal_len=15,
                        d_state=16, num_classes=5).to(device)

    logits = model(dummy_input)
    Y_hat = torch.softmax(logits, dim=-1)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {Y_hat.shape}")
    print(f"Contains NaN: {torch.isnan(Y_hat).any().item()}")
    print(f"Total num parameters: {sum(p.numel() for p in model.parameters()):,}")
