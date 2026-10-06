import torch
import torch.nn as nn

import utils
from models.conditions import ObjectiveConditionEncoder, TimestepEmbedder
from models.layers import Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Denoiser(nn.Module):
    def __init__(
        self,
        max_n_nodes,
        hidden_size=384,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        drop_condition=0.1,
        Xdim=118,
        Edim=5,
        Posdim=8,
        ydim=1,
        task_type="regression",
        condition_mode="none",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.ydim = ydim
        self.pos_dim = Posdim
        self.condition_mode = str(condition_mode).strip().lower()

        token_hidden = hidden_size // 2
        edge_hidden = hidden_size // 4
        pos_hidden = hidden_size - token_hidden - edge_hidden

        self.token_embedder = nn.Linear(Xdim, token_hidden, bias=False)
        self.bond_embedder = nn.Linear(max_n_nodes * Edim, edge_hidden, bias=False)
        self.pos_embedder = nn.Linear(max_n_nodes * Posdim, pos_hidden, bias=False)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.cond_encoder = None
        if self.condition_mode == "multiobjective_vector" and ydim > 0:
            self.cond_encoder = ObjectiveConditionEncoder(hidden_size, ydim, drop_condition)

        self.encoders = nn.ModuleList([SELayer(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.out_layer = OutLayer(
            max_n_nodes=max_n_nodes,
            hidden_size=hidden_size,
            token_hidden=token_hidden,
            edge_hidden=edge_hidden,
            pos_hidden=pos_hidden,
            atom_type=Xdim,
            bond_type=Edim,
            pos_dim=Posdim,
        )
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        def _constant_init(module, value):
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, value)
                if module.bias is not None:
                    nn.init.constant_(module.bias, value)

        self.apply(_basic_init)
        for block in self.encoders:
            _constant_init(block.adaLN_modulation[0], 0)
        _constant_init(self.out_layer.adaLN_modulation[0], 0)

    def forward(self, x, e, pos, node_mask, y, t, unconditioned, return_features=False, deterministic_policy=False):
        x_in = x
        e_in = e
        pos_in = pos
        bs, n, _ = x.size()
        x_feat = self.token_embedder(x)
        e_feat = self.bond_embedder(e.reshape(bs, n, -1))
        pos_feat = self.pos_embedder(pos.reshape(bs, n, -1))
        x = torch.cat([x_feat, e_feat, pos_feat], dim=-1)

        c = self.t_embedder(t)
        if self.cond_encoder is not None:
            if y is None or y.numel() == 0:
                y = torch.zeros((x.size(0), self.ydim), device=x.device, dtype=x.dtype)
                drop_ids = torch.ones(x.size(0), device=x.device, dtype=torch.bool) if unconditioned else None
            else:
                y = y.to(device=x.device, dtype=x.dtype)
                if y.dim() == 1:
                    y = y.view(x.size(0), self.ydim)
                drop_ids = torch.isnan(y).all(dim=1)
                if unconditioned:
                    drop_ids = torch.ones_like(drop_ids)
            cond = self.cond_encoder(y, self.training and not deterministic_policy, drop_ids)
            c = c + cond

        for block in self.encoders:
            x = block(x, c, node_mask)

        X, E, pos = self.out_layer(x, x_in, e_in, pos_in, c, node_mask)
        out = utils.PlaceHolder(X=X, E=E, y=None, pos=pos).mask(node_mask)
        if return_features:
            return out, x
        return out


class SELayer(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.dropout = 0.0
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), drop=self.dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c, node_mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * modulate(self.norm1(self.attn(x, node_mask=node_mask)), shift_msa, scale_msa)
        x = x + gate_mlp.unsqueeze(1) * modulate(self.norm2(self.mlp(x)), shift_mlp, scale_mlp)
        return x


class OutLayer(nn.Module):
    def __init__(self, max_n_nodes, hidden_size, token_hidden, edge_hidden, pos_hidden, atom_type, bond_type, pos_dim):
        super().__init__()
        self.atom_type = atom_type
        self.bond_type = bond_type
        self.pos_dim = pos_dim
        self.token_hidden = token_hidden
        self.edge_hidden = edge_hidden
        self.pos_hidden = pos_hidden

        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.linear_x = nn.Linear(token_hidden, atom_type, bias=False)
        self.linear_e = nn.Linear(edge_hidden, max_n_nodes * bond_type, bias=False)
        self.linear_pos = nn.Linear(pos_hidden, max_n_nodes * pos_dim, bias=False)

    def forward(self, x, x_in, e_in, pos_in, c, node_mask):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        B, N, _ = x.size()
        x_tok = x[:, :, : self.token_hidden]
        x_edge = x[:, :, self.token_hidden : self.token_hidden + self.edge_hidden]
        x_pos = x[:, :, self.token_hidden + self.edge_hidden :]

        atom_out = self.linear_x(x_tok) + x_in
        bond_out = self.linear_e(x_edge).reshape(B, N, N, self.bond_type) + e_in
        pos_out = self.linear_pos(x_pos).reshape(B, N, N, self.pos_dim) + pos_in

        edge_mask = (~node_mask)[:, :, None] & (~node_mask)[:, None, :]
        diag_mask = torch.eye(N, dtype=torch.bool, device=bond_out.device).unsqueeze(0).expand(B, -1, -1)
        bond_out.masked_fill_(edge_mask[:, :, :, None], 0)
        bond_out.masked_fill_(diag_mask[:, :, :, None], 0)
        pos_out.masked_fill_(edge_mask[:, :, :, None], 0)
        pos_out.masked_fill_(diag_mask[:, :, :, None], 0)
        bond_out = 0.5 * (bond_out + torch.transpose(bond_out, 1, 2))
        return atom_out, bond_out, pos_out
