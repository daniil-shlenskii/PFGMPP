import random

import numpy as np
import torch


def set_seed(seed: int=None):
    if seed is None:
        return
    # Python
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch (CPU and CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU case
