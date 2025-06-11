from copy import deepcopy
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import LongTensor, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from pfgmpp.core.pfgmpp import PFGMPP
from pfgmpp.utils.nn import get_device_from_net


class ClassifierTrainer:
    def __init__(
        self,
        *,
        pfgmpp: PFGMPP,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
    ):
        self.pfgmpp = pfgmpp
        self.net = net
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = get_device_from_net(net)

    def train(
        self,
        *,
        train_loader: DataLoader,
        n_iters: int,
        log_every: int = 100,
        save_path: Optional[str] = None,
        verbose: bool = True,
    ):
        pbar = tqdm(train_loader, total=n_iters, dynamic_ncols=True, colour="green", disable=not verbose)
        acc_batch_loss = 0.
        for i, (x, label) in enumerate(pbar):
            batch_loss = self._train_step(x=x, label=label)
            acc_batch_loss += batch_loss / log_every
            if (i + 1) % log_every == 0:
                pbar.set_postfix(loss=acc_batch_loss)
                acc_batch_loss = 0.
            if i == n_iters:
                break
        if save_path is not None:
            torch.save(self.net.state_dict(), save_path)

    def _train_step(self, *, x: Tensor, label: LongTensor):
        self.net.train()

        x, label = x.to(self.device), label.to(self.device)

        loss = self.loss_fn(net=self.net, x=x, label=label).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
