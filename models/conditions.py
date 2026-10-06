import math

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t = t.view(-1)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class ObjectiveConditionEncoder(nn.Module):
    def __init__(self, hidden_size, ydim, dropout_prob):
        super().__init__()
        self.ydim = int(ydim)
        self.dropout_prob = float(dropout_prob)
        if self.dropout_prob > 0:
            self.drop_embedding = nn.Embedding(1, hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(self.ydim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, y, train, force_drop_ids=None):
        if y is None or y.numel() == 0:
            y = torch.zeros((0, self.ydim), device=self.proj[0].weight.device, dtype=self.proj[0].weight.dtype)
        if y.dim() == 1:
            y = y.view(-1, self.ydim)
        if y.size(1) != self.ydim:
            raise ValueError(f"Expected conditioning vector with dim {self.ydim}, got {y.size(1)}.")
        nan_rows = torch.isnan(y).all(dim=1)
        cond = self.proj(torch.nan_to_num(y.float(), nan=0.0))
        if self.dropout_prob <= 0:
            return cond
        if force_drop_ids is None:
            drop_ids = nan_rows.to(cond.device)
        else:
            drop_ids = torch.logical_or(force_drop_ids.view(-1).bool(), nan_rows.to(cond.device))
        if train:
            rand_drop = torch.rand(cond.size(0), device=cond.device) < self.dropout_prob
            drop_ids = torch.logical_or(drop_ids, rand_drop)
        if drop_ids.any():
            cond = cond.clone()
            cond[drop_ids] = self.drop_embedding.weight[0]
        return cond
