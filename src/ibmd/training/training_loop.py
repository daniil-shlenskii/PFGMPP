from typing import Callable, Optional

import torch.nn as nn
from hydra.utils import instantiate
from tqdm import tqdm

from ibmd.core.ibmd import IBMD


def training_loop(
    *,
    batch_size: int,
    inner_problem_iters: int,
    n_iters: int,
    #
    teacher_net: nn.Module,
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
    callback: Optional[Callable] = None,
    save_path: Optional[str] = None,
    verbose: bool = True,
    #
    device: str = "cpu",
):
    teacher_dynamics = instantiate(teacher_dynamics_config)
    # teacher_net = instantiate(teacher_net_config).to(device)
    teacher_loss_fn = instantiate(teacher_loss_fn_config, **{teacher_loss_dynamics_key: teacher_dynamics})
    ibmd = IBMD(
        teacher_dynamics=teacher_dynamics,
        teacher_net=teacher_net,
        teacher_loss_fn=teacher_loss_fn,
        student_net_optimizer_config=student_net_optimizer_config,
        student_data_estimator_net_config=student_data_estimator_net_optimizer_config,
        n_classes=n_classes,
        ema_decay=ema_decay,
    )
    training_loop_instantiated(
        ibmd=ibmd,
        #
        batch_size=batch_size,
        inner_problem_iters=inner_problem_iters,
        n_iters=n_iters,
        #
        log_every=log_every,
        eval_every=eval_every,
        callback=callback,
        save_path=save_path,
        verbose=verbose,
    )

def training_loop_instantiated(
    *,
    ibmd: IBMD,
    #
    batch_size: int,
    inner_problem_iters: int,
    n_iters: int,
    #
    log_every: int = 100,
    eval_every: int = 100,
    callback: Optional[Callable] = None,
    save_path: Optional[str] = None,
    verbose: bool = True,
):
    pbar = tqdm(range(n_iters), total=n_iters, dynamic_ncols=True, colour="green", disable=not verbose)
    for it in pbar:
        batch_loss = ibmd.train_step(batch_size=batch_size, inner_problem_iters=inner_problem_iters)

        if it == 0:
            acc_batch_loss = batch_loss
        else:
            acc_batch_loss = (acc_batch_loss * it + batch_loss) / (it + 1)

        if (it + 1) % log_every == 0:
            pbar.set_postfix(student_loss=acc_batch_loss)
            acc_batch_loss = 0.
        if callback is not None and (it + 1) % eval_every == 0:
            callback(ibmd, it=it)
        if it == n_iters:
            break

    if save_path is not None:
        ibmd.save(save_path)
