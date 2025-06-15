from torch.nn.parallel import DistributedDataParallel as DDP

from ibmd.core.ibmd import IBMD
from ibmd.nn.utils import get_device_from_net


class IBMD_DDP(IBMD):
    def __init__(self, *, rank: int=0, local_rank: int=0, **kwargs):
        self.rank = rank
        self.local_rank = local_rank
        super().__init__(**kwargs)

    def _setup(self):
        student_net, student_net_ema, student_data_estimator_net = super()._setup()
        if get_device_from_net(self.teacher_net).type != "cpu":
            student_net = DDP(student_net, device_ids=[self.local_rank])
            student_data_estimator_net = DDP(student_data_estimator_net, device_ids=[self.local_rank])
        return student_net, student_net_ema, student_data_estimator_net

    def save(self, save_path: str):
        if self.rank == 0:
            super().save(save_path)
