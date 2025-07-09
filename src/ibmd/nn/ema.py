from copy import deepcopy
from typing import Dict

import torch

from ibmd.nn.utils import freeze_model, get_device_from_net


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.device = get_device_from_net(model)

        self.model = model.module if hasattr(model, 'module') else model
        self.decay = decay

        self.ema = deepcopy(model).to(self.device).eval()
        freeze_model(self.ema)

    @torch.no_grad()
    def update(self):
        for ema_param, model_param in zip(
            self.ema.parameters(), self.model.parameters()
        ):
            ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1-self.decay)

    def state_dict(self) -> Dict:
        return {
            "ema_state_dict": self.ema.state_dict(),
            "model_state_dict": self.model.state_dict(),
            "decay": self.decay,
        }

    def load_state_dict(self, state_dict):
        self.ema.load_state_dict(state_dict["ema_state_dict"])
        self.model.load_state_dict(state_dict["model_state_dict"])
        self.decay = state_dict.get("decay", self.decay)
