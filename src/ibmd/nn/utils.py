import torch.nn as nn


def get_device_from_net(net: nn.Module) -> str:
    return next(net.parameters()).device

def freeze_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
