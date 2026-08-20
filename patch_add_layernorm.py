path = "src/models/sleepmamba.py"
with open(path, "r") as f:
    content = f.read()

old_init = '''        self.dam_layers = nn.ModuleList([
            DualAxisMambaBlock(latent_dim=latent_dim, temporal_len=temporal_len, d_state=d_state)
            for _ in range(n_dam_layers)
        ])
        self.sbm_layers = nn.ModuleList([
            SequenceBiMamba(latent_dim=latent_dim, d_state=d_state)
            for _ in range(n_sbm_layers)
        ])'''

new_init = '''        self.dam_layers = nn.ModuleList([
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
        self.sbm_norms = nn.ModuleList([nn.LayerNorm(latent_dim) for _ in range(n_sbm_layers)])'''

content = content.replace(old_init, new_init)

old_forward = '''        Z = F_pp
        for i, dam in enumerate(self.dam_layers):
            if i < len(self.dam_layers) - 1:
                Z = dam.forward_no_pool(Z)
            else:
                Z = dam(Z)

        g = Z
        G = g.view(B, T, self.D)

        O = G
        for sbm in self.sbm_layers:
            O = sbm(O)'''

new_forward = '''        Z = F_pp
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
            O = self.sbm_norms[i](O)'''

content = content.replace(old_forward, new_forward)

with open(path, "w") as f:
    f.write(content)
print("Patched.")
