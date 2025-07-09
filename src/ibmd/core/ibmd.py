from copy import deepcopy
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch import LongTensor

from ibmd.core.multistep import get_timestamps
from ibmd.nn.ema import ModelEMA
from ibmd.nn.utils import get_device_from_net, remove_dropout_from_model


class IBMD:
    def __init__(
        self,
        *,
        # teacher
        teacher_dynamics: Any,
        teacher_net: nn.Module,
        teacher_loss_fn: Callable,
        # student data estimator
        student_data_estimator_net_config: dict,
        # student
        student_net_optimizer_config: dict,
        teacher_loss_fn_for_student: Callable = None,
        student_sample_timestamps: tuple[float] = None,
        # sampling
        n_steps: int = 1,
        sampling_mode: str = "uniform",
        #
        n_classes: int = None,
        ema_decay: float = 0.999,
        remove_dropout: bool = False,
    ):
        self.teacher_dynamics = teacher_dynamics
        self.teacher_net = teacher_net
        self.teacher_loss_fn = teacher_loss_fn
        self.teacher_loss_fn_for_student = (
            teacher_loss_fn_for_student
            if teacher_loss_fn_for_student is not None
            else teacher_loss_fn
        )

        self.n_classes = n_classes
        self.ema_decay = ema_decay
        self.remove_dropout = remove_dropout
        self.device = get_device_from_net(teacher_net)

        (
            self.student_net,
            self.student_net_ema,
            self.student_data_estimator_net,
        ) = self._setup()

        self.student_net_optimizer = torch.optim.Adam(
            params=self.student_net.parameters(),
            **student_net_optimizer_config
        )
        self.student_data_estimator_net_optimizer = torch.optim.Adam(
            params=self.student_data_estimator_net.parameters(),
            **student_data_estimator_net_config,
        )

        self.sampling_timestamps = get_timestamps(
            mode=sampling_mode,
            n_steps=n_steps,
            sigma_min=self.teacher_dynamics.sigma_min,
            sigma_max=self.teacher_dynamics.sigma_max,
        )

    def _setup(self):
        student_net = deepcopy(self.teacher_net).to(self.device)
        student_data_estimator_net = deepcopy(self.teacher_net).to(self.device)
        student_net_ema = ModelEMA(model=student_net, decay=self.ema_decay)

        if self.remove_dropout:
            remove_dropout_from_model(student_net)
            remove_dropout_from_model(student_data_estimator_net)

        return student_net, student_net_ema, student_data_estimator_net

    @torch.no_grad()
    def sample(self, *, sample_size: int, label: Optional[LongTensor]=None, seed: int=None):
        x_noisy = self.teacher_dynamics.sample_from_prior(sample_size, seed=seed).to(self.device)
        for i, t in enumerate(self.sampling_timestamps):
            t = torch.full((sample_size,), t).to(self.device)
            x = self.student_net_ema.ema(x=x_noisy, t=t, label=label)
            if i < len(self.sampling_timestamps) - 1:
                t_next = self.sampling_timestamps[i + 1]
                t_next = torch.full((sample_size,), t_next).to(self.device)
                x_noisy = self.teacher_dynamics.sample_from_posterior(x=x, t=t_next, seed=seed)
        return x

    def sample_from_student(self, sample_size: int, label: Optional[LongTensor]=None, seed: int=None):
        x_noisy = self.teacher_dynamics.sample_from_prior(sample_size, seed=seed).to(self.device)
        for i, t in enumerate(self.sampling_timestamps):
            t = torch.full((sample_size,), t).to(self.device)
            x = self.student_net(x=x_noisy, t=t, label=label)
            if i < len(self.sampling_timestamps) - 1:
                t_next = self.sampling_timestamps[i + 1]
                t_next = torch.full((sample_size,), t_next).to(self.device)
                x_noisy = self.teacher_dynamics.sample_from_posterior(x=x, t=t_next, seed=seed)
        return x

    def train_step(self, *, batch_size, inner_problem_iters: int):
        for _ in range(inner_problem_iters):
            _ = self.update_student_data_estimator(batch_size=batch_size)
        student_net_loss = self.update_student(batch_size=batch_size)
        return student_net_loss

    def update_student(self, *, batch_size: int):
        self.student_net.train()
        self.student_data_estimator_net.eval()
        self.teacher_net.eval()

        label = None
        if self.n_classes is not None:
            label = torch.randint(0, self.n_classes, (batch_size,)).to(self.device)

        shared_seed = torch.randint(0, 2**32, (1,)).item()
        student_batch = self.sample_from_student(sample_size=batch_size, label=label)
        teacher_loss = self.teacher_loss_fn_for_student(net=self.teacher_net, x=student_batch, label=label, seed=shared_seed)
        student_data_estimator_loss = self.teacher_loss_fn_for_student(net=self.student_data_estimator_net, x=student_batch, label=label, seed=shared_seed)
        loss = (teacher_loss - student_data_estimator_loss).mean()

        self.student_net_optimizer.zero_grad()
        loss.backward()
        self.student_net_optimizer.step()

        self.student_net_ema.update()

        return loss.item()

    def update_student_data_estimator(self, *, batch_size: int):
        self.student_net.eval()
        self.student_data_estimator_net.train()

        label = None
        if self.n_classes is not None:
            label = torch.randint(0, self.n_classes, (batch_size,)).to(self.device)

        with torch.no_grad():
            student_batch = self.sample_from_student(sample_size=batch_size, label=label)
        loss = self.teacher_loss_fn(
            net=self.student_data_estimator_net, x=student_batch, label=label
        ).mean()

        self.student_data_estimator_net_optimizer.zero_grad()
        loss.backward()
        self.student_data_estimator_net_optimizer.step()

        return loss.item()

    def save(self, save_path: str):
        torch.save(self.student_net_ema.state_dict(), save_path)

    def load(self, load_path: str):
        self.student_net_ema.load_state_dict(torch.load(load_path))
        self.student_net.load_state_dict(self.student_net_ema.model.state_dict())
