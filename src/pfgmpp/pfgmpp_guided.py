import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import LongTensor, Tensor
from tqdm import tqdm

from utils.data import get_inifinite_loader

from .pfgmpp import PFGMPP


class PFGMPPGuided(PFGMPP):
    def __init__(
        self,
        cls: nn.Module,
        cls_optimizer: torch.optim.Optimizer,
        cls_sigma_prior_mode: str = "sqrt",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cls = cls
        self.cls_optimizer = cls_optimizer
        self.cls_sample_from_sigma_prior = self._get_sigma_prior(cls_sigma_prior_mode)

    @torch.no_grad()
    def sample(self, *, label: LongTensor, guidance_scale: float, **kwargs):
        self.cls.eval()
        return super().sample(label=label, guidance_scale=guidance_scale, **kwargs)

    @torch.no_grad()
    def sample_unconditional(self, **kwargs):
        self.cls.eval()
        return self.sample(guidance_scale=0., **kwargs)

    def drift(self, x_hat: Tensor, t: Tensor, label: LongTensor, guidance_scale: float=0., **kwargs):
        uncoditional_drift = self.model.drift(x_hat=x_hat, t=t, D=self.D, label=label)
        classifer_score = (
            self.classifier_score(x_hat=x_hat, t=t, label=label)
            if label is not None
            else torch.zeros_like(x_hat)
        )
        return uncoditional_drift + classifer_score * t * guidance_scale

    def classifier_score(self, *, x_hat: Tensor, t: Tensor, label: LongTensor): 
        with torch.enable_grad():
            x_hat.requires_grad_(True)
            logits = self.cls(x=x_hat, t=t)
            log_probs = F.log_softmax(logits, dim=-1)
            class_log_probs = log_probs[torch.arange(len(logits)), label.view(-1)]
            # print(log_probs.exp()[0].detach().numpy(), class_log_probs[0].exp().item(), label[0].item())
            return torch.autograd.grad(class_log_probs.sum(), x_hat)[0]

    def train_classifier(self, train_loader: torch.utils.data.DataLoader, n_iters: int, verbose: bool=True, log_every: int=100):
        train_loader = get_inifinite_loader(train_loader)
        pbar = tqdm(train_loader, total=n_iters, dynamic_ncols=True, colour="green", disable=not verbose)
        acc_batch_loss, acc_batch_score = 0., 0.
        acc_sigma_min_loss, acc_sigma_mean_loss, acc_sigma_max_loss = 0., 0., 0.
        acc_sigma_min_score, acc_sigma_mean_score, acc_sigma_max_score = 0., 0., 0.
        for i, (x, label) in enumerate(pbar):
            batch_loss, batch_score = self._train_classifier_step(x=x, label=label)
            acc_batch_loss += batch_loss / log_every
            acc_batch_score += batch_score / log_every

            # eval
            sigma_min_loss, sigma_min_score = self._eval_at_t(x=x, t=self.sigma_min, label=label)
            sigma_mean_loss, sigma_mean_score = self._eval_at_t(x=x, t=(self.sigma_min + self.sigma_max), label=label)
            sigma_max_loss, sigma_max_score = self._eval_at_t(x=x, t=self.sigma_max, label=label)
            acc_sigma_min_loss += sigma_min_loss / log_every
            acc_sigma_min_score += sigma_min_score / log_every
            acc_sigma_mean_loss += sigma_mean_loss / log_every
            acc_sigma_mean_score += sigma_mean_score / log_every
            acc_sigma_max_loss += sigma_max_loss / log_every
            acc_sigma_max_score += sigma_max_score / log_every
            if verbose and (i == 0 or (i + 1) % log_every == 0):
                pbar.set_postfix(
                    loss=acc_batch_loss,
                    score=acc_batch_score,
                    sigma_min_loss=acc_sigma_min_loss,
                    sigma_mean_loss=acc_sigma_mean_loss,
                    sigma_max_loss=acc_sigma_max_loss,
                    sigma_min_score=acc_sigma_min_score,
                    sigma_mean_score=acc_sigma_mean_score,
                    sigma_max_score=acc_sigma_max_score
                )
                acc_batch_loss, acc_batch_score = 0., 0.
                acc_sigma_min_loss, acc_sigma_mean_loss, acc_sigma_max_loss = 0., 0., 0.
                acc_sigma_min_score, acc_sigma_mean_score, acc_sigma_max_score = 0., 0., 0.
            if i == n_iters:
                break

    def _train_classifier_step(self, *, x: Tensor, label: LongTensor):
        self.cls.train()
        x, label = x.to(self.device), label.to(self.device)
        t = self.cls_sample_from_sigma_prior(x.shape[0]).to(self.device)
        x_hat = self.sample_from_posterior(x=x, t=t)
        logits = self.cls(x=x_hat, t=t)
        loss = (F.cross_entropy(logits, label)).mean()

        self.cls_optimizer.zero_grad()
        loss.backward()
        self.cls_optimizer.step()

        batch_loss = loss.item()
        batch_score = torch.eq(torch.argmax(logits, dim=1), label).to(torch.float32).mean().item()
        return batch_loss, batch_score

    @torch.no_grad()
    def _eval_at_t(self, x: Tensor, t: float, label: LongTensor):
        self.cls.eval()
        t = torch.full((len(x),), t).to(x.device)
        x_hat = self.sample_from_posterior(x=x, t=t)
        logits = self.cls(x=x_hat, t=t)
        loss = F.cross_entropy(logits, label).mean().item()
        score = torch.eq(torch.argmax(logits, dim=1), label).to(torch.float32).mean().item()
        return loss, score

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.cls.state_dict(), os.path.join(save_dir, "cls.pt"))
        super().save(save_dir=save_dir)

    def load(self, load_dir: str):
        cls_path = os.path.join(load_dir, "cls.pt")
        if not os.path.exists(cls_path):
            logger.warning(f"{cls_path} does not exist")
        else:
            self.cls.load_state_dict(torch.load(cls_path))
        super().load(load_dir=load_dir)
