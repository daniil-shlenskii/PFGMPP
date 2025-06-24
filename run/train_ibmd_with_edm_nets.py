import os
import pickle
import sys
from typing import Any, Callable, Optional

import hydra
import torch
import torch.distributed as dist
import torch.nn as nn
from constants import DNNLIB_DIR, EDM_UTILS_DIR, TORCH_UTILS_DIR
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from ibmd.training.training_loop import training_loop_instantiated


class EDMNetWrapper(nn.Module):
    def __init__(
        self,
        *,
        edm_net: nn.Module,
        img_channels: int,
        img_resolution: int,
        n_classes: int=None
    ):
        super().__init__()
        self.edm_net = edm_net

        self.img_channels = img_channels
        self.img_resolution = img_resolution
        self.n_classes = n_classes

        self.data_dim = self.img_channels * self.img_resolution**2

    def forward(self, x: Tensor, t: Tensor, label: LongTensor=None):
        # modify input
        x = x.reshape(-1, self.img_channels, self.img_resolution, self.img_resolution)
        if label is None:
            class_labels = None
        else:
            class_labels = F.one_hot(label, num_classes=self.n_classes)

        # apply edm net
        out = self.edm_net(x=x, sigma=t, class_labels=class_labels)

        # modify output
        return out.reshape(-1, self.data_dim)

def load_edm_net(ckpt_path: str) -> nn.Module:
    edm_utils_dirs = [DNNLIB_DIR, EDM_UTILS_DIR, TORCH_UTILS_DIR]
    sys.path.extend(edm_utils_dirs)
    with open(ckpt_path, "rb") as f:
        edm_net = pickle.load(f)['ema'].to("cpu");
    sys.path = sys.path[:-len(edm_utils_dirs)]
    return edm_net

def training_loop(
    *,
    run_dir: str,
    #
    batch_size: int,
    inner_problem_iters: int,
    n_iters: int,
    #
    img_channels: int,
    img_resolution: int,
    teacher_net_ckpt_path: str,
    teacher_dynamics_config: dict,
    teacher_loss_fn_config: dict,
    student_net_optimizer_config: dict,
    student_data_estimator_net_optimizer_config: dict,
    teacher_loss_dynamics_key: str = "pfgmpp",
    n_classes: int = None,
    ema_decay: float = 0.999,
    #
    log_every: int = 500,
    eval_every: int = 500,
    callback: Optional[dict] = None,
    verbose: bool = True,
):
    edm_net = load_edm_net(teacher_net_ckpt_path)
    teacher_net = EDMNetWrapper(
        edm_net=edm_net,
        img_channels=img_channels,
        img_resolution=img_resolution,
        n_classes=n_classes,
    )
    teacher_dynamics = instantiate(teacher_dynamics_config)
    teacher_loss_fn = instantiate(teacher_loss_fn_config, **{teacher_loss_dynamics_key: teacher_dynamics})

    if callback is not None:
        callback = instantiate(callback)

    training_loop_instantiated(
        run_dir=run_dir,
        #
        teacher_dynamics=teacher_dynamics,
        teacher_net=teacher_net,
        teacher_loss_fn=teacher_loss_fn,
        student_net_optimizer_config=student_net_optimizer_config,
        student_data_estimator_net_optimizer_config=student_data_estimator_net_optimizer_config,
        n_classes=n_classes,
        ema_decay=ema_decay,
        #
        batch_size=batch_size,
        inner_problem_iters=inner_problem_iters,
        n_iters=n_iters,
        #
        log_every=log_every,
        eval_every=eval_every,
        callback=callback,
        verbose=verbose,
    )

@hydra.main(config_path="configs", config_name="ibmd_edm")
def main(config: DictConfig):
    training_loop(
        img_channels=config.ibmd.teacher.img_channels,
        img_resolution=config.ibmd.teacher.img_resolution,
        teacher_net_ckpt_path=config.ibmd.teacher.ckpt_path,
        teacher_dynamics_config=config.ibmd.teacher.dynamics,
        teacher_loss_fn_config=config.ibmd.teacher.loss,
        student_net_optimizer_config=config.ibmd.student.net_optimizer,
        student_data_estimator_net_optimizer_config=config.ibmd.student.data_estimator_net_optimizer,
        ema_decay=config.ibmd.ema_decay,
        n_classes=config.ibmd.n_classes,
        **config.train,
        **config.eval,
    )

if __name__ == "__main__":
    main()
