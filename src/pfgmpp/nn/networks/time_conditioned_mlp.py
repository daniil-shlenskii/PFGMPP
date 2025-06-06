from typing import Optional

import torch.nn as nn
from torch import Tensor


class TimeConditionedMLP(nn.Module):
    def __init__(self, dim, hidden_dim, n_layers, out_dim=None, n_classes: int=None, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.input_proj = nn.Linear(dim, hidden_dim)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
            for _ in range(n_layers)
        ])

        out_dim = dim if out_dim is None else out_dim
        self.output_proj = nn.Linear(hidden_dim, out_dim)

        if n_classes:
            self.label_embed = nn.Sequential(
                nn.Embedding(n_classes, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def forward(self, *, x: Tensor, t: Tensor, label: Optional[Tensor]=None) -> Tensor:
        t_embed = self.time_embed(t.view(-1, 1))
        if self.n_classes:
            label_embed = self.label_embed(label.view(-1))
            t_embed = t_embed + label_embed
        h = self.input_proj(x) + t_embed
        for layer in self.layers:
            h = layer(h)
        return self.output_proj(h)
