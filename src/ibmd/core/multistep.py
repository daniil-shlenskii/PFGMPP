import numpy as np


def get_timestamps(mode: str, n_steps: int, sigma_min: float, sigma_max: float):
    if mode == "uniform":
        timestamps = np.linspace(sigma_min, sigma_max, n_steps + 1)[1:][::-1]
    else:
        raise NotImplementedError
    return timestamps
