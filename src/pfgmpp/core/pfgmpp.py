import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch import LongTensor, Tensor
from tqdm import tqdm
from typing_extensions import override

from utils.data import get_inifinite_loader

EPS = 1e-8


class PFGMPP:
    def __init__(
        self,
        *,
        data_dim: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        #
        sigma_prior_mode: str = "log_normal",
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        D: int = 1,
        #
        device: str = "cpu",
    ):
        super().__init__()
        self.data_dim = data_dim
        self.model = model
        self.optimizer = optimizer
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.D = D
        self.device = device

        self.sample_from_sigma_prior = self._get_sigma_prior(sigma_prior_mode)

    @torch.no_grad()
    def sample(self, *, sample_size: int, num_steps: int=32, rho: float=7.0, label: Optional[LongTensor]=None, **drift_kwargs):
        assert label is None or len(label) == sample_size
        self.model.eval()

        # Noise samling
        x_next = self.sample_from_prior(sample_size)

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float, device=self.device)
        sigma_steps = (self.sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (self.sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))) ** rho
        sigma_steps = torch.cat([torch.as_tensor(sigma_steps), torch.zeros_like(sigma_steps[:1])]).to(self.device).float() # t_N = 0

        # Sampling with Euler scheme
        for t_cur, t_next in zip(sigma_steps[:-1], sigma_steps[1:]):
            x_cur = x_next
            x_next = x_cur + self.drift(x_hat=x_cur, t=t_cur, label=label, **drift_kwargs) * (t_next - t_cur)

        return x_next

    @override
    def drift(self, *, x_hat: Tensor, t: Tensor, label: Optional[LongTensor]=None, **kwargs):
        return self.model.drift(x_hat=x_hat, t=t, D=self.D, label=label)

    def sample_from_posterior(self, *, x: Tensor, t: Tensor):
        r = t * self.D**0.5
        # Sampling form inverse-beta distribution
        samples_norm = np.random.beta(
            a=self.data_dim / 2., b=self.D / 2.,
            size=x.shape[0],
        ).astype(np.double)
        inverse_beta = samples_norm / (1 - samples_norm + EPS)
        inverse_beta = torch.from_numpy(inverse_beta).to(self.device).double()
        # Sampling from p_r(R) by change-of-variable (c.f. Appendix B)
        R = (r.squeeze(-1) * np.sqrt(inverse_beta + EPS)).view(len(x), -1)
        # Uniformly sample the angle component
        gaussian = torch.randn(len(x), self.data_dim).to(R.device)
        unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
        # Construct the perturbation
        return x + (unit_gaussian * R).float()

    def sample_from_prior(self, sample_size: int):
        x = torch.zeros(sample_size, self.data_dim).to(self.device)
        t = torch.full((sample_size,), self.sigma_max).to(self.device)
        return self.sample_from_posterior(x=x, t=t)

    def train(self, *, train_loader: torch.utils.data.DataLoader, n_iters: int, verbose: bool=True, log_every: int=100):
        train_loader = get_inifinite_loader(train_loader)
        pbar = tqdm(train_loader, total=n_iters, dynamic_ncols=True, colour="green", disable=not verbose)
        acc_batch_loss = 0.
        for i, (x, label) in enumerate(pbar):
            batch_loss = self._train_step(x=x, label=label)
            acc_batch_loss += batch_loss / log_every
            if i == 0 or (i + 1) % log_every == 0:
                pbar.set_postfix(loss=acc_batch_loss)
                acc_batch_loss = 0.
            if i == n_iters:
                break

    def _train_step(self, *, x: Tensor, label: LongTensor=None):
        self.model.train()
        x, label = x.to(self.device), label.to(self.device)
        t = self.sample_from_sigma_prior(x.shape[0]).to(self.device)
        x_hat = self.sample_from_posterior(x=x, t=t)
        loss = self.model.loss(t=t, x=x, x_hat=x_hat, label=label).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _get_sigma_prior(self, mode: str):
        if mode == "log_normal":
            def sample_from_sigma_prior(sample_size: int):
                rnd_normal = torch.randn((sample_size, 1), device=self.device)
                sigma = (rnd_normal * 1.2 - 1.2).exp()
                return torch.clip(sigma, min=self.sigma_min, max=self.sigma_max)
        elif mode == "uniform":
            def sample_from_sigma_prior(sample_size: int):
                return torch.rand((sample_size, 1), device=self.device)
        elif mode == "linear":
            def sample_from_sigma_prior(sample_size: int):
                a, b = self.sigma_min, self.sigma_max
                u = torch.rand(sample_size)
                samples = a + (b - a) * torch.sqrt(u)
                return samples
        elif mode == "sqrt":
            def sample_from_sigma_prior(sample_size: int):
                a, b = self.sigma_min, self.sigma_max
                u = torch.rand(sample_size)
                samples = a + (b - a) * u.pow(2/3)
                return samples
        else:
            raise NotImplementedError
        return sample_from_sigma_prior

    def save_model(self, save_path: str):
        torch.save(self.model.state_dict(),save_path)

    def load_model(self, load_path: str):
        if not os.path.exists(load_path):
            logger.warning(f"{load_path} does not exist")
        else:
            self.model.load_state_dict(torch.load(load_path))
