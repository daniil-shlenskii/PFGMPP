import torch.nn as nn


def get_device_from_net(net: nn.Module) -> str:
    return next(net.parameters()).device
