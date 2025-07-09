import torch
from torch import Tensor


def get_loss_weights(*, mode: str, sigma_min: float, sigma_max: float):
    if mode == "uniform":
        def loss_weights(t: Tensor):
            return torch.ones_like(t)
    elif mode == "edm":
        def loss_weights(t: Tensor, sigma_data: float=0.5):
            return (t**2 + sigma_data) / (t**2 + sigma_data**2)
    elif mode == "exp":
        def loss_weights(t: Tensor):
            return (sigma_max - t).exp()
    elif mode == "exp_inv":
        def loss_weights(t: Tensor):
            return (t - sigma_min).exp()
    else:
        raise NotImplementedError

    return loss_weights
