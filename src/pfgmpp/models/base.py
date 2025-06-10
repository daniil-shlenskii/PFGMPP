from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from pfgmpp.nn.networks import TimeConditionedMLP


class BaseField(nn.Module):
    def __init__(
        self,
        dim=2,
        hidden_dim=128,
        n_layers=4,
        n_classes=None,
        sigma_data=0.5,
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.net = TimeConditionedMLP(
            dim=dim, hidden_dim=hidden_dim, n_layers=n_layers, n_classes=n_classes
        )

    def forward(self, *, x: Tensor, t: Tensor, label: Optional[Tensor]=None) -> Tensor:
        return self.net(x=x, t=t, label=label)

    def drift(self, *, x_hat: Tensor, t: Tensor, D: int, label: Optional[Tensor]=None):
       return (x_hat - self(x=x_hat, t=t, label=label)) / t

    def loss(self, *, t: Tensor, x: Tensor, x_hat: Tensor, label: Optional[Tensor]=None):
        loss = torch.sum((self(x=x_hat, t=t, label=label) - x)**2, dim=1) * self.weight(t).view(-1, 1) 
        return loss

    def weight(self, t: Tensor):
        return (t**2 + self.sigma_data) / (t**2 + self.sigma_data**2)
