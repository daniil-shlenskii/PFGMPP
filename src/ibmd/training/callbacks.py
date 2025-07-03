import abc
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from cleanfid import fid

from ibmd.core import IBMD


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
        mode: str = "clean"
    ):
        self.img_channels = img_channels
        self.img_resolution = img_resolution
        self.n_classes = n_classes

        self.dataset_name = dataset_name
        self.dataset_split = dataset_split

        self.num_samples_per_class = num_samples_per_class
        self.batch_size = batch_size

        self.mode = mode

    def __call__(self, ibmd: IBMD, it: int, eval_dir: str, seed: int=0):
        fake_images = []
        for class_idx in range(self.n_classes):
            collected_images = 0
            for i in range(0, self.num_samples_per_class, self.batch_size):
                batch_size = min(self.batch_size, self.num_samples_per_class - collected_images)
                labels = torch.full((batch_size,), class_idx, dtype=torch.long).to(ibmd.device)
                batch = ibmd.sample(
                    sample_size=batch_size, label=labels.to(ibmd.device), seed=i,
                ).reshape(-1, self.img_channels, self.img_resolution, self.img_resolution).cpu().numpy()
                batch = (batch.clip(-1, 1) * 127.5 + 127.5).astype(np.uint8)  # [-1, 1] -> [0, 255]
                fake_images.append(batch)
                collected_images += len(batch)
        fake_images = np.concatenate(fake_images, axis=0)

        # Save generated images temporarily (clean-fid requires images on disk)
        temp_dir = f"{eval_dir}/fid_temp_{it}"
        os.makedirs(temp_dir, exist_ok=True)
        for i, img in enumerate(fake_images):
            plt.imsave(f"{temp_dir}/{i}.png", img.transpose(1, 2, 0))

        # Compute FID
        score = fid.compute_fid(
            temp_dir,
            dataset_name=self.dataset_name,
            dataset_res=self.img_resolution,
            dataset_split=self.dataset_split,
            device=ibmd.device,
            batch_size=self.batch_size,
            mode=self.mode,
        )
        with open(f"{eval_dir}/fid.txt", "a") as f:
            f.write(f"{it}: {score:.3f}\n")

        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
