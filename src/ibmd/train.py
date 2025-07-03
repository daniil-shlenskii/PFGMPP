import hydra
from omegaconf import DictConfig

from ibmd.training.training_loop import training_loop


@hydra.main(config_path="configs", config_name="ibmd")
def main(config: DictConfig):
    training_loop(
        run_dir=config.train.run_dir,
        #
        batch_size=config.train.batch_size,
        inner_problem_iters=config.train.inner_problem_iters,
        n_iters=config.train.n_iters,
        #
        teacher_net_config=config.ibmd.teacher.net,
        teacher_net_ckpt_path=config.ibmd.teacher.ckpt_path,
        teacher_dynamics_config=config.ibmd.teacher.dynamics,
        teacher_loss_fn_config=config.ibmd.teacher.loss,
        teacher_loss_fn_for_student_config=config.ibmd.teacher.loss_for_student,
        student_net_optimizer_config=config.ibmd.student.net_optimizer,
        student_data_estimator_net_optimizer_config=config.ibmd.student.data_estimator_net_optimizer,
        ema_decay=config.ibmd.ema_decay,
        n_classes=config.ibmd.n_classes,
        #
        log_every=config.eval.log_every,
        eval_every=config.eval.eval_every,
        callbacks=config.eval.get("callbacks", {}),
        verbose=config.eval.get("verbose", True),
    )

if __name__ == "__main__":
    main()
