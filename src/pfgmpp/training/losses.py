import abc
from typing import Optional

import torch
import torch.nn as nn
from torch import LongTensor, Tensor

from pfgmpp.core import PFGMPP
from pfgmpp.training.utils.loss_weights import get_loss_weights
from pfgmpp.training.utils.sigma_prior import get_sigma_prior


class PFGMPPLoss(abc.ABC):
    def __init__(
        self,
        *,
        pfgmpp: PFGMPP,
        loss_weights_mode: str,
        sigma_prior_mode: str,
    ):
        self.pfgmpp = pfgmpp
        self.loss_weights = get_loss_weights(
            mode=loss_weights_mode,
            sigma_min=pfgmpp.sigma_min,
            sigma_max=pfgmpp.sigma_max,
        )
        self.sample_from_sigma_prior = get_sigma_prior(
            mode=sigma_prior_mode,
            sigma_min=pfgmpp.sigma_min,
            sigma_max=pfgmpp.sigma_max,
        )

    @abc.abstractmethod
    def __call__(self, *, net: nn.Module, x: Tensor, label: Optional[LongTensor]=None, seed: Optional[int]=None):
        pass

class TargetPredictionLoss(PFGMPPLoss):
    def __init__(
        self,
        *,
        pfgmpp: PFGMPP,
        loss_weights_mode: str = "edm",
        sigma_prior_mode: str = "log_normal",
    ):
        super().__init__(
            pfgmpp=pfgmpp,
            loss_weights_mode=loss_weights_mode,
            sigma_prior_mode=sigma_prior_mode
        )

    def __call__(self, *, net: nn.Module, x: Tensor, label: Optional[LongTensor]=None, seed: Optional[int]=None):
        t = self.sample_from_sigma_prior(x.shape[0], seed=seed).to(x.device)
        x_hat = self.pfgmpp.sample_from_posterior(x=x, t=t, seed=seed)
        D_x = net(x=x_hat, t=t, label=label)

        loss_batchwise = torch.sum((D_x - x)**2, dim=1)
        return loss_batchwise * self.loss_weights(t).view(-1, 1)
