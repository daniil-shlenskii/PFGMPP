import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from cleanfid import fid

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
    def __init__(
        self,
        # dataset params
        img_channels: int,
        img_resolution: int,
        # visualization params
        n_classes: int,
        sample_size_per_class: int,
        # fid params
        dataset_name: str,
        dataset_split: str,
        fid_num_samples: int,
        fid_batch_size: int,
    ):
        self.img_channels = img_channels
        self.img_resolution = img_resolution

        self.n_classes = n_classes
        self.sample_size_per_class = sample_size_per_class
        self.labels = torch.tensor([
            [i] * sample_size_per_class for i in range(n_classes)
        ]).long().view(-1)
        self.sample_size = n_classes * sample_size_per_class

        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.fid_num_samples = fid_num_samples
        self.fid_batch_size = fid_batch_size

    def __call__(self, ibmd: IBMD, it: int, eval_dir: str, seed: int=0):
        self.save_visualization(ibmd, it, eval_dir, seed)
        self.compute_fid(ibmd, eval_dir, it, seed)

    def save_visualization(self, ibmd: IBMD, it: int, eval_dir: str, seed: int=0):
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

    def compute_fid(self, ibmd, eval_dir: str, it: int, seed: int = 0):
        # Generate samples for FID (normalized to [0, 255] uint8)
        fake_images = []
        for _ in range(0, self.fid_num_samples, self.fid_batch_size):
            batch_size = min(self.fid_batch_size, self.fid_num_samples - len(fake_images))
            z = torch.randn(batch_size, ibmd.latent_dim).to(ibmd.device)
            labels = torch.randint(0, self.n_classes, (batch_size,)).to(ibmd.device)
            batch = ibmd.sample(z, labels).cpu().numpy()
            batch = (batch * 127.5 + 127.5).astype(np.uint8)  # [-1, 1] -> [0, 255]
            fake_images.append(batch)
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
            batch_size=self.fid_batch_size,
            mode="clean",
        )
        with open(f"{eval_dir}/fid.txt", "a") as f:
            f.write(f"{it}: {score:.3f}\n")

        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
