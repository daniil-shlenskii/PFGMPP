from copy import deepcopy
from typing import Dict

import torch

from ibmd.nn.utils import freeze_model, get_device_from_net


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay

        self.device = get_device_from_net(model)

        self.ema = deepcopy(model).eval()
        self.ema.to(self.device)
        freeze_model(self.ema)

    @torch.no_grad()
    def update(self):
        for ema_param, model_param in zip(
            self.ema.parameters(), self.model.parameters()
        ):
            ema_param.data =\
                ema_param.data * self.decay + model_param.data * (1 - self.decay)

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
