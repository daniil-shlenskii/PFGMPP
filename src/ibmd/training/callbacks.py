import matplotlib.pyplot as plt
import numpy as np
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

    def __call__(self, ibmd: IBMD, it: int, eval_dir: str, seed: int=0):
        gens = ibmd.sample(sample_size=self.sample_size, label=self.labels.to(ibmd.device)).cpu().numpy()
        plt.figure(figsize=(4, 4))
        sns.scatterplot(x=gens[:, 0], y=gens[:, 1], hue=self.labels)
        plt.xlabel(""); plt.ylabel("")
        plt.savefig(f"{eval_dir}/{it}.png")
        plt.close()

class ImageDataCallback:
    def __init__(self, n_classes: int, sample_size_per_class: int, img_channels: int, img_resolution: int):
        self.n_classes = n_classes
        self.sample_size_per_class = sample_size_per_class
        self.img_channels = img_channels
        self.img_resolution = img_resolution

        self.labels = torch.tensor([
            [i] * sample_size_per_class for i in range(n_classes)
        ]).long().view(-1)
        self.sample_size = n_classes * sample_size_per_class

    def __call__(self, ibmd: IBMD, it: int, eval_dir: str, seed: int=0):
        gens = ibmd.sample(
            sample_size=self.sample_size, label=self.labels.to(ibmd.device), seed=seed,
        ).reshape(-1, self.img_channels, self.img_resolution, self.img_resolution).cpu().numpy()

        # normalization
        gens -= np.min(gens, axis=0)
        gens /= np.max(gens, axis=0)

        ncols = 4
        nrows = (self.sample_size + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
        fig.set_figheight(nrows * 2)
        fig.set_figwidth(ncols * 2)

        for ax_idx in range(nrows * ncols):
            i, j = ax_idx // ncols, ax_idx % ncols
            ax = axes[i, j]
            if ax_idx < len(gens):
                ax.axis("off")
                ax.imshow(gens[ax_idx].transpose(1, 2, 0));
            else:
                fig.delaxes(ax)
        plt.savefig(f"{eval_dir}/{it}.png")
        plt.close()
