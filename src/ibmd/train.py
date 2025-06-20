import hydra
from omegaconf import DictConfig

from ibmd.training.training_loop import training_loop


@hydra.main(config_path="configs", config_name="ibmd")
def main(config: DictConfig):
    training_loop(
        teacher_net_config=config.ibmd.teacher.net,
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
