import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, Tensor

from pfgmpp.core import PFGMPP
from pfgmpp.training.utils.sigma_prior import get_sigma_prior


class ClassifierLoss:
    def __init__(
        self,
        *,
        pfgmpp: PFGMPP,
        sigma_prior_mode: str = "sqrt",
    ):
        self.pfgmpp = pfgmpp
        self.sample_from_sigma_prior = get_sigma_prior(
            mode=sigma_prior_mode,
            sigma_min=pfgmpp.sigma_min,
            sigma_max=pfgmpp.sigma_max,
        )

    def __call__(self, *, net: nn.Module, x: Tensor, label: LongTensor):
        t = self.sample_from_sigma_prior(x.shape[0]).to(x.device)
        x_hat = self.pfgmpp.sample_from_posterior(x=x, t=t)
        logits = net(x=x_hat, t=t)
        loss_batchwise = F.cross_entropy(logits, label)
        return loss_batchwise
