from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor

from pfgmpp.utils.reproducibility import set_seed

EPS = 1e-8


class PFGMPP:
    def __init__(
        self,
        *,
        data_dim: int,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        D: int = 1,
    ):
        self.data_dim = data_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.D = D

    def sample_from_posterior(self, *, x: Tensor, t: Tensor, seed: Optional[int]=None):
        set_seed(seed)

        r = t * self.D**0.5
        # Sampling form inverse-beta distribution
        samples_norm = np.random.beta(
            a=self.data_dim / 2., b=self.D / 2.,
            size=x.shape[0],
        ).astype(np.double)
        inverse_beta = samples_norm / (1 - samples_norm + EPS)
        inverse_beta = torch.from_numpy(inverse_beta).to(x.device).double()
        # Sampling from p_r(R) by change-of-variable (c.f. Appendix B)
        R = (r.squeeze(-1) * np.sqrt(inverse_beta + EPS)).view(len(x), -1)
        # Uniformly sample the angle component
        gaussian = torch.randn(len(x), self.data_dim).to(R.device)
        unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
        # Construct the perturbation
        return x + (unit_gaussian * R).float()

    def sample_from_prior(self, sample_size: int, seed: Optional[int]=None):
        x = torch.zeros(sample_size, self.data_dim)
        t = torch.full((sample_size,), self.sigma_max)
        return self.sample_from_posterior(x=x, t=t, seed=seed)

    @torch.no_grad()
    def sample(
        self,
        *,
        drift: Callable,
        sample_size: int,
        num_steps: int = 32,
        rho: float = 7.0,
        label: Optional[LongTensor] = None,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        assert label is None or len(label) == sample_size
        set_seed(seed)

        # Noise samling
        x_next = self.sample_from_prior(sample_size)

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float, device=device)
        sigma_steps = (self.sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (self.sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))) ** rho
        sigma_steps = torch.cat([torch.as_tensor(sigma_steps), torch.zeros_like(sigma_steps[:1])]).to(device).float() # t_N = 0

        # Sampling with Euler scheme
        for t_cur, t_next in zip(sigma_steps[:-1], sigma_steps[1:]):
            x_cur = x_next
            x_next = x_cur + drift(x=x_cur, t=t_cur, label=label) * (t_next - t_cur)

        return x_next

    @torch.no_grad()
    def sample_with_classifier(
        self,
        *,
        drift: Callable,
        classifier: Callable,
        guidance_scale: float,
        **kwargs,
    ):
        def classifier_score(x: Tensor, t: Tensor, label: LongTensor):
            with torch.enable_grad():
                x.requires_grad_(True)
                logits = classifier(x=x, t=t)
                log_probs = F.log_softmax(logits, dim=-1)
                class_log_probs = log_probs[torch.arange(len(logits)), label.view(-1)]
                return torch.autograd.grad(class_log_probs.sum(), x)[0]
        def guided_drift(x: Tensor, t: Tensor, label: LongTensor):
            return drift(x=x, t=t, label=label) - classifier_score(x=x, t=t, label=label) * guidance_scale * t
        return self.sample(drift=guided_drift, **kwargs)
