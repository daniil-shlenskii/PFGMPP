from typing import Callable, Optional

from tqdm import tqdm

from ibmd.core.ibmd import IBMD


def training_loop_instantiated(
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
