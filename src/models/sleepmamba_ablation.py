"""
sleepmamba_ablation.py

Model variants for the component ablation study (paper Table 7):
- SleepMambaAblation(ablation="none")       : full model (baseline)
- SleepMambaAblation(ablation="no_sdam")     : skip SDAM in MLE
- SleepMambaAblation(ablation="no_dam_intra"): DAM uses only inter-modal branch
- SleepMambaAblation(ablation="no_dam_inter"): DAM uses only intra-modal branch
- SleepMambaAblation(ablation="no_sbm")       : skip Sequence Bi-Mamba entirely

Keeps LayerNorm + input clamping (our confirmed NaN fixes) in all variants.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from mle import MultimodalLocalEncoder, MultiScaleCNN
from dam import DualAxisMambaBlock
from sequence_bi_mamba import SequenceBiMamba


class MLENoSDAM(nn.Module):
    def __init__(self, in_channels, latent_dim=128, dropout=0.5):
        super().__init__()
        self.mscnn = MultiScaleCNN(in_channels, dropout=dropout)
        self.project = nn.Sequential(
            nn.Conv1d(256, latent_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        fused = self.mscnn(x)
        F = self.project(fused)
        return F


class DualAxisMambaBlockAblation(nn.Module):
    def __init__(self, latent_dim, temporal_len, d_state=16, mode="full"):
        super().__init__()
        self.D = latent_dim
        self.E = temporal_len
        self.mode = mode

        self.gate_proj = nn.Linear(self.E, self.E)
        if mode in ("full", "intra_only"):
            self.ssm_temporal = DualAxisMambaBlock(latent_dim, temporal_len, d_state).ssm_temporal
        if mode in ("full", "inter_only"):
            self.ssm_modal = DualAxisMambaBlock(latent_dim, temporal_len, d_state).ssm_modal

    def forward_no_pool(self, F_double_prime):
        V_i = torch.sigmoid(self.gate_proj(F_double_prime))

        if self.mode in ("full", "intra_only"):
            temp_in = F_double_prime.transpose(1, 2)
            temp_out = self.ssm_temporal(temp_in)
            temp_out = temp_out.transpose(1, 2)
            Z_intra = temp_out * V_i
        else:
            Z_intra = 0

        if self.mode in ("full", "inter_only"):
            mod_out = self.ssm_modal(F_double_prime)
            Z_inter = mod_out * V_i
        else:
            Z_inter = 0

        Z_i = Z_intra + Z_inter
        return Z_i

    def forward(self, F_double_prime):
        Z_i = self.forward_no_pool(F_double_prime)
        return Z_i.mean(dim=2) # type: ignore


class SleepMambaAblation(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, temporal_len=15,
                 d_state=16, num_classes=5, n_dam_layers=2, n_sbm_layers=2,
                 dropout=0.5, ablation="none"):
        super().__init__()
        self.D = latent_dim
        self.E = temporal_len
        self.ablation = ablation

        if ablation == "no_sdam":
            self.mle = MLENoSDAM(in_channels, latent_dim, dropout)
        else:
            self.mle = MultimodalLocalEncoder(in_channels, latent_dim, dropout)

        dam_mode = "full"
        if ablation == "no_dam_intra":
            dam_mode = "inter_only"
        elif ablation == "no_dam_inter":
            dam_mode = "intra_only"

        self.dam_layers = nn.ModuleList([
            DualAxisMambaBlockAblation(latent_dim, temporal_len, d_state, mode=dam_mode)
            for _ in range(n_dam_layers)
        ])
        self.dam_norms = nn.ModuleList([nn.LayerNorm(temporal_len) for _ in range(n_dam_layers - 1)])
        self.post_dam_norm = nn.LayerNorm(latent_dim)

        self.use_sbm = (ablation != "no_sbm")
        if self.use_sbm:
            self.sbm_layers = nn.ModuleList([
                SequenceBiMamba(latent_dim, d_state) for _ in range(n_sbm_layers)
            ])
            self.sbm_norms = nn.ModuleList([nn.LayerNorm(latent_dim) for _ in range(n_sbm_layers)])

        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, X):
        X = torch.clamp(X, min=-10.0, max=10.0)
        B, T, C, L = X.shape
        X_flat = X.view(B * T, C, L)
        F_pp = self.mle(X_flat)

        Z = F_pp
        for i, dam in enumerate(self.dam_layers):
            if i < len(self.dam_layers) - 1:
                Z = dam.forward_no_pool(Z) # type: ignore
                Z = self.dam_norms[i](Z)
            else:
                Z = dam(Z)

        g = self.post_dam_norm(Z)
        G = g.view(B, T, self.D)

        if self.use_sbm:
            O = G
            for i, sbm in enumerate(self.sbm_layers):
                O = sbm(O)
                O = self.sbm_norms[i](O)
        else:
            O = G

        logits = self.classifier(O)
        return logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, C, L = 2, 5, 3, 3000
    dummy_input = torch.randn(B, T, C, L).to(device)

    for ablation in ["none", "no_sdam", "no_dam_intra", "no_dam_inter", "no_sbm"]:
        model = SleepMambaAblation(in_channels=C, ablation=ablation).to(device)
        logits = model(dummy_input)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{ablation:15s} | output={tuple(logits.shape)} | "
              f"NaN={torch.isnan(logits).any().item()} | params={n_params:,}")