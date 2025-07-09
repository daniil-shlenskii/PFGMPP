import abc

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from cleanfid import fid

from ibmd.core import IBMD
from ibmd.utils.reproducibility import set_seed


class IBMDCallback(abc.ABC):
    @abc.abstractmethod
    def __call__(self, ibmd: IBMD, it: int, eval_dir: str, seed: int=0):
        pass

class CallbacksHandler(IBMDCallback):
    def __init__(self, callbacks: list = None):
        self.callbacks = callbacks or []

    def __call__(self, ibmd: IBMD, it: int, eval_dir: str, seed: int=0):
        for callback in self.callbacks:
            callback(ibmd, it, eval_dir, seed)


class TwoDimCallback(IBMDCallback):
    def __init__(self, sample_size: int):
        self.sample_size = sample_size
        self.labels = torch.concatenate([
            torch.zeros(sample_size//2), torch.ones(sample_size//2),
        ]).to(torch.long)

    def __call__(self, ibmd: IBMD, it: int, eval_dir: str, seed: int=0):
        gens = ibmd.sample(sample_size=self.sample_size, label=self.labels.to(ibmd.device)).cpu().numpy()
        plt.figure(figsize=(4, 4))
        sns.scatterplot(x=gens[:, 0], y=gens[:, 1], hue=self.labels)
        plt.xlabel("")
        plt.ylabel("")
        plt.savefig(f"{eval_dir}/{it}.png")
        plt.close()


class VisualizeImageModelCallback(IBMDCallback):
    def __init__(
        self,
        img_channels: int,
        img_resolution: int,
        n_classes: int,
        sample_size_per_class: int,
    ):
        self.img_channels = img_channels
        self.img_resolution = img_resolution

        self.n_classes = n_classes
        self.sample_size_per_class = sample_size_per_class
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
                ax.imshow(gens[ax_idx].transpose(1, 2, 0))
            else:
                fig.delaxes(ax)
        plt.savefig(f"{eval_dir}/{it}.png")

class FIDCallback(IBMDCallback):
    def __init__(
        self,
        img_channels: int,
        img_resolution: int,
        n_classes: int,
        #
        dataset_name: str,
        dataset_split: str,
        #
        num_samples_per_class: int,
        batch_size: int,
        #
        mode: str = "clean",
    ):
        self.img_channels = img_channels
        self.img_resolution = img_resolution
        self.n_classes = n_classes

        self.dataset_name = dataset_name
        self.dataset_split = dataset_split

        self.num_samples_per_class = num_samples_per_class
        self.batch_size = batch_size

        self.mode = mode

        self.total_samples = n_classes * num_samples_per_class
        self.label_schedule = torch.cat([
            torch.full((num_samples_per_class,), cls)
            for cls in range(n_classes)
        ])

    def __call__(self, ibmd: IBMD, it: int, eval_dir: str, seed: int=0):
        set_seed(seed)

        batch_start_idx = 0

        @torch.no_grad()
        def sample_fn(unused_clean_fid_latent: torch.Tensor):
            nonlocal batch_start_idx
            labels = self.label_schedule[batch_start_idx : batch_start_idx + self.batch_size].to(ibmd.device)
            batch_size = labels.shape[0]
            batch_start_idx += batch_size
            images = ibmd.sample(
                sample_size=batch_size, label=labels.to(ibmd.device),
            ).reshape(-1, self.img_channels, self.img_resolution, self.img_resolution)
            images = images.clamp(-1, 1) * 127.5 + 127.5
            return images.to(torch.uint8)

        score = fid.compute_fid(
            gen=sample_fn,
            dataset_name=self.dataset_name,
            dataset_res=self.img_resolution,
            dataset_split=self.dataset_split,
            device=ibmd.device,
            batch_size=self.batch_size,
            mode=self.mode,
        )

        with open(f"{eval_dir}/fid.txt", "a") as f:
            f.write(f"{it}: {score:.3f}\n")
