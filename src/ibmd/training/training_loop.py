import os
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from hydra.utils import instantiate
from tqdm import tqdm

from ibmd.core.ibmd import IBMD
from ibmd.core.ibmd_ddp import IBMD_DDP
from ibmd.training.callbacks import CallbacksHandler


def setup(backend="auto"):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    return rank, local_rank, world_size, device

def is_ddp():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def training_loop(
    *,
    run_dir: str,
    #
    batch_size: int,
    inner_problem_iters: int,
    n_iters: int,
    #
    teacher_net_config: dict,
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
    callbacks: Optional[dict | list[dict]] = None,
    verbose: bool = True,
):
    teacher_net = instantiate(teacher_net_config)
    teacher_net.load_state_dict(torch.load(teacher_net_ckpt_path, map_location="cpu"))
    teacher_dynamics = instantiate(teacher_dynamics_config)
    teacher_loss_fn = instantiate(teacher_loss_fn_config, **{teacher_loss_dynamics_key: teacher_dynamics})

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
        callbacks=callbacks,
        verbose=verbose,
    )

def training_loop_instantiated(
    *,
    run_dir: str,
    #
    teacher_dynamics: Any,
    teacher_net: nn.Module,
    teacher_loss_fn: Callable,
    student_net_optimizer_config: dict,
    student_data_estimator_net_optimizer_config: dict,
    n_classes: int,
    ema_decay: float,
    #
    batch_size: int,
    inner_problem_iters: int,
    n_iters: int,
    #
    log_every: int = 500,
    eval_every: int = 500,
    callbacks: Optional[Callable] = None,
    verbose: bool = True,
):
    # setup run dir
    ckpt_path = os.path.join(run_dir, "ckpt.pt")
    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    if is_ddp():
        # setup multiprocess training
        rank, local_rank, world_size, device = setup()
        effective_batch_size = batch_size // world_size

        ibmd = IBMD_DDP(
            teacher_dynamics=teacher_dynamics,
            teacher_net=teacher_net.to(device),
            teacher_loss_fn=teacher_loss_fn,
            student_net_optimizer_config=student_net_optimizer_config,
            student_data_estimator_net_config=student_data_estimator_net_optimizer_config,
            n_classes=n_classes,
            ema_decay=ema_decay,
            rank=rank,
            local_rank=local_rank,
        )
    else:
        # setup single gpu training
        rank = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        effective_batch_size = batch_size

        ibmd = IBMD(
            teacher_dynamics=teacher_dynamics,
            teacher_net=teacher_net.to(device),
            teacher_loss_fn=teacher_loss_fn,
            student_net_optimizer_config=student_net_optimizer_config,
            student_data_estimator_net_config=student_data_estimator_net_optimizer_config,
            n_classes=n_classes,
            ema_decay=ema_decay,
        )

    pbar = tqdm(range(n_iters), total=n_iters, dynamic_ncols=True, colour="green", disable=not verbose or rank != 0)
    for it in pbar:
        batch_loss = ibmd.train_step(batch_size=effective_batch_size, inner_problem_iters=inner_problem_iters)

        if it == 0:
            acc_batch_loss = batch_loss
        else:
            acc_batch_loss = (acc_batch_loss * it + batch_loss) / (it + 1)

        if it % log_every == 0:
            pbar.set_postfix(student_loss=acc_batch_loss)
            acc_batch_loss = 0.
        if rank == 0 and callbacks is not None and it % eval_every == 0:
            callbacks(ibmd, it=it, eval_dir=eval_dir)
            ibmd.save(ckpt_path)
        if it == n_iters:
            break

    ibmd.save(ckpt_path)
