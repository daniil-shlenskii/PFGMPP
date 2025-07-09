import os
import pickle
import sys
from typing import Optional

import hydra
import torch.nn as nn
import torch.nn.functional as F
from constants import (ARTIFACTS_DIR, CHECKPOINTS_DIR, DNNLIB_DIR,
                       EDM_UTILS_DIR, TORCH_UTILS_DIR)
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import LongTensor, Tensor

from ibmd.training.callbacks import CallbacksHandler
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
        x = x.view(-1, self.img_channels, self.img_resolution, self.img_resolution)
        if label is None:
            class_labels = None
        else:
            class_labels = F.one_hot(label, num_classes=self.n_classes)

        # apply edm net
        out = self.edm_net(x=x, sigma=t, class_labels=class_labels)

        # modify output
        return out.view(-1, self.data_dim)

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
    # Teacher
    n_classes: int,
    img_channels: int,
    img_resolution: int,
    teacher_net_ckpt_path: str,
    teacher_dynamics_config: dict,
    teacher_loss_fn_config: dict,
    # Student data estimator
    student_data_estimator_net_optimizer_config: dict,
    # Student
    student_net_optimizer_config: dict,
    teacher_loss_fn_for_student_config: dict | None,
    # Other kwargs
    ibmd_kwargs: dict,
    # Training 
    batch_size: int,
    inner_problem_iters: int,
    n_iters: int,
    # Evaluation
    log_every: int = 500,
    eval_every: int = 500,
    callbacks: Optional[dict | list[dict]] = None,
    verbose: bool = True,
    #
    teacher_loss_dynamics_key: str = "pfgmpp",
):
    edm_net = load_edm_net(teacher_net_ckpt_path)
    teacher_net = EDMNetWrapper(
        edm_net=edm_net,
        img_channels=img_channels,
        img_resolution=img_resolution,
        n_classes=n_classes,
    )
    for p in teacher_net.parameters():
        p.requires_grad = True
    teacher_dynamics = instantiate(teacher_dynamics_config)
    teacher_loss_fn = instantiate(teacher_loss_fn_config, **{teacher_loss_dynamics_key: teacher_dynamics})
    if teacher_loss_fn_for_student_config is not None:
        teacher_loss_fn_for_student = instantiate(teacher_loss_fn_for_student_config, **{teacher_loss_dynamics_key: teacher_dynamics})
    else:
        teacher_loss_fn_for_student = None 

    if callbacks is not None:
        callbacks = CallbacksHandler(
            callbacks=[instantiate(callback) for callback in callbacks]
        )

    training_loop_instantiated(
        run_dir=run_dir,
        #
        teacher_dynamics=teacher_dynamics,
        teacher_net=teacher_net,
        teacher_loss_fn=teacher_loss_fn,
        #
        student_data_estimator_net_optimizer_config=student_data_estimator_net_optimizer_config,
        #
        student_net_optimizer_config=student_net_optimizer_config,
        teacher_loss_fn_for_student=teacher_loss_fn_for_student,
        #
        ibmd_kwargs=ibmd_kwargs,
        #
        batch_size=batch_size,
        inner_problem_iters=inner_problem_iters,
        n_iters=n_iters,
        #
        log_every=log_every,
        eval_every=eval_every,
        callbacks=callbacks,
        verbose=verbose,
    )

@hydra.main(config_path="configs", config_name="ibmd_edm")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    training_loop(
        run_dir=os.path.join(ARTIFACTS_DIR, config.train.run_dir),
        #
        # Teacher
        n_classes=config.ibmd.teacher.n_classes,
        img_channels=config.ibmd.teacher.img_channels,
        img_resolution=config.ibmd.teacher.img_resolution,
        teacher_net_ckpt_path=os.path.join(CHECKPOINTS_DIR, config.ibmd.teacher.ckpt_path),
        teacher_dynamics_config=config.ibmd.teacher.dynamics,
        teacher_loss_fn_config=config.ibmd.teacher.loss,
        # Student data estimator
        student_data_estimator_net_optimizer_config=config.ibmd.student.data_estimator_net_optimizer,
        # Student
        student_net_optimizer_config=config.ibmd.student.net_optimizer,
        teacher_loss_fn_for_student_config=config.ibmd.teacher.get("loss_for_student"),
        #
        ibmd_kwargs=config.ibmd.other,
        # Training
        batch_size=config.train.batch_size,
        inner_problem_iters=config.train.inner_problem_iters,
        n_iters=config.train.n_iters,
        # Evaluation
        log_every=config.eval.log_every,
        eval_every=config.eval.eval_every,
        callbacks=config.eval.get("callbacks", {}),
        verbose=config.eval.get("verbose", True),
    )

if __name__ == "__main__":
    main()
