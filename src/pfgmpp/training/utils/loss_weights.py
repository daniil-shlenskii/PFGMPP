import torch
from torch import Tensor


def get_loss_weights(*, mode: str):
    if mode == "uniform":
        def loss_weights(t: Tensor):
            return torch.ones_like(t)
    elif mode == "edm":
        def loss_weights(t: Tensor, sigma_data: float=0.5):
            return (t**2 + sigma_data) / (t**2 + sigma_data**2)
    else:
        raise NotImplementedError

    return loss_weights
