import abc
from typing import Optional

import torch
import torch.nn as nn
from torch import LongTensor, Tensor

from pfgmpp.core import PFGMPP
from pfgmpp.training.utils.sigma_prior import get_sigma_prior


class PFGMPPLoss(abc.ABC):
    def __init__(
        self,
        *,
        pfgmpp: PFGMPP,
        sigma_prior_mode: str = "log_normal",
    ):
        self.pfgmpp = pfgmpp
        self.sample_from_sigma_prior = get_sigma_prior(sigma_prior_mode)

    @abc.abstractmethod
    def __call__(self, net: nn.Module, x: Tensor, label: Optional[LongTensor]=None):
        pass

class EDMLoss(PFGMPPLoss):
    def __init__(
        self,
        *,
        pfgmpp: PFGMPP,
        sigma_prior_mode: str = "log_normal",
        sigma_data: float = 0.5,
    ):
        super().__init__(pfgmpp=pfgmpp, sigma_prior_mode=sigma_prior_mode)
        self.sigma_data = sigma_data

    def __call__(self, net: nn.Module, x: Tensor, label: Optional[LongTensor]=None):
        t = self.sample_from_sigma_prior(x.shape[0]).to(x.device)
        x_hat = self.pfgmpp.sample_from_posterior(x=x, t=t)
        D_x = net(x=x_hat, t=t, label=label)

        loss_batchwise = torch.sum((D_x - x)**2, dim=1)
        weight = (t**2 + self.sigma_data) / (t**2 + self.sigma_data**2)
        return loss_batchwise * weight.view(-1, 1)
