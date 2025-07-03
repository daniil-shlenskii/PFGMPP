import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, Tensor


def get_device_from_net(net: nn.Module) -> str:
    return next(net.parameters()).device

def freeze_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False

def remove_dropout_from_model(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            # Need to get parent module to replace the dropout
            *parent_names, child_name = name.split('.')
            parent = model
            for pname in parent_names:
                parent = getattr(parent, pname)
            setattr(parent, child_name, nn.Identity())
