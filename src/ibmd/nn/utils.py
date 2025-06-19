import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, Tensor


def get_device_from_net(net: nn.Module) -> str:
    return next(net.parameters()).device

def freeze_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False

class EDMNetWrapper(nn.Module):
    def __init__(
        self,
        *,
        edm_net: nn.Module,
        img_channels: int,
        img_resolution: int,
        n_classes: int=None
    ):
        super().__init__()
        self.edm_net = edm_net

        self.img_channels = img_channels
        self.img_resolution = img_resolution
        self.n_classes = n_classes

        self.data_dim = self.img_channels * self.img_resolution**2

    def forward(self, x: Tensor, t: Tensor, label: LongTensor=None):
        # modify input
        x = x.reshape(-1, self.img_channels, self.img_resolution, self.img_resolution)
        if label is None:
            class_labels = None
        else:
            class_labels = F.one_hot(label, num_classes=self.n_classes)

        # apply edm net
        out = self.edm_net(x=x, sigma=t, class_labels=class_labels)

        # modify output
        return out.reshape(-1, self.data_dim)
