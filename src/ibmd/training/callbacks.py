import matplotlib.pyplot as plt
import seaborn as sns
import torch
from IPython.display import clear_output

from ibmd.core import IBMD


class TwoDimCallback:
    def __init__(self, sample_size: int):
        self.sample_size = sample_size
        self.labels = torch.concatenate([
            torch.zeros(sample_size//2), torch.ones(sample_size//2),
        ]).to(torch.long)

    def __call__(self, ibmd: IBMD, it: int, device: torch.device, eval_dir: str):
        clear_output()
        gens = ibmd.sample(sample_size=self.sample_size, label=self.labels.to(device)).cpu().numpy()
        plt.figure(figsize=(4, 4))
        sns.scatterplot(x=gens[:, 0], y=gens[:, 1], hue=self.labels)
        plt.xlabel(""); plt.ylabel("")
        plt.savefig(f"{eval_dir}/{it}.png")
        plt.close()
