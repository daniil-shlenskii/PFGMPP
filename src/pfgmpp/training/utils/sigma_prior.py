import functools
from typing import Optional

import torch

from pfgmpp.utils.reproducibility import set_seed


def get_sigma_prior(*, mode: str, sigma_min: float, sigma_max: float):
    if mode == "log_normal":
        def sample_from_sigma_prior(sample_size: int):
            rnd_normal = torch.randn(sample_size)
            sigma = (rnd_normal * 1.2 - 1.2).exp()
            return torch.clip(sigma, min=sigma_min, max=sigma_max)
    elif mode == "uniform":
        def sample_from_sigma_prior(sample_size: int):
            return torch.rand(sample_size)
    elif mode == "linear":
        def sample_from_sigma_prior(sample_size: int):
            a, b = sigma_min, sigma_max
            u = torch.rand(sample_size)
            samples = a + (b - a) * torch.sqrt(u)
            return samples
    elif mode == "sqrt":
        def sample_from_sigma_prior(sample_size: int):
            a, b = sigma_min, sigma_max
            u = torch.rand(sample_size)
            samples = a + (b - a) * u.pow(2/3)
            return samples
    else:
        raise NotImplementedError

    def reproducibility_decorator(func):
        @functools.wraps(func)
        def wrapper(sample_size, seed: Optional[int]=None):
            set_seed(seed)
            return func(sample_size)
        return wrapper
    return reproducibility_decorator(sample_from_sigma_prior)
