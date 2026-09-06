"""
dataset.py
==========
Turns the per-layer physics from `optics.py` into training data.

Two datasets are provided:

1. SyntheticFlakes: generates labelled microscope-style scenes on the fly .

2. RealFlakes: an images/folder of RGB pictures
   and a masks/folder of single-channel PNGs whose pixel values ARE the layer
   count (0,1,2,...). Use this once you have annotated microscope images. uSe any color by
   number too online.
   
"""

from __future__ import annotations
import numpy as np
from PIL import Image, ImageFilter

import torch
from torch.utils.data import Dataset

from optics import color_table


# --------------------------------------------------------------------------- #
#  Shared normalisation (single source of truth for train AND inference)       #
# --------------------------------------------------------------------------- #
def normalize_img(img_hwc: np.ndarray) -> np.ndarray:
    """Map an RGB image in [0,1] to roughly [-1,1]."""
    return (img_hwc.astype(np.float32) - 0.5) / 0.5


def to_chw_tensor(img_hwc: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img_hwc.transpose(2, 0, 1)))


# --------------------------------------------------------------------------- #
#  Synthetic scene generation                                                  #
# --------------------------------------------------------------------------- #
def _blob_field(rng, H, W, n_centres):
    """Smooth field in [0,1] from a sum of rotated 2D Gaussians."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    f = np.zeros((H, W), np.float32)
    for _ in range(n_centres):
        cy, cx = rng.uniform(0.15, 0.85) * H, rng.uniform(0.15, 0.85) * W
        sy, sx = rng.uniform(0.06, 0.20) * H, rng.uniform(0.06, 0.20) * W
        th = rng.uniform(0, np.pi)
        xr = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
        yr = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
        f += np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    f /= f.max() + 1e-9
    return f


def _illumination(rng, H, W):
    """Low-frequency multiplicative lighting field + radial vignette."""
    base = 1.0 + 0.12 * (_blob_field(rng, H, W, n_centres=2) - 0.5) * 2.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    vignette = 1.0 - 0.18 * np.clip(r, 0, 1) ** 2
    return (base * vignette).astype(np.float32)


def make_scene(rng, size, max_layers, colors):
    """
    Build one synthetic scene.

    Returns
    -------
    img   : float32 HxWx3 in [0,1]
    label : int64   HxW    (0..max_layers)
    """
    H = W = size
    label = np.zeros((H, W), np.int64)

    # 1..3 flakes, each a stack of terraces via level sets of a blob field.
    for _ in range(rng.integers(1, 4)):
        g = _blob_field(rng, H, W, n_centres=rng.integers(1, 3))
        n_terr = int(rng.integers(1, max_layers + 1))
        # thresholds: higher field -> more layers (nested terraces)
        thr = np.sort(rng.uniform(0.30, 0.90, size=n_terr))[::-1]
        flake = np.zeros((H, W), np.int64)
        for t in thr:
            flake += (g > t).astype(np.int64)
        flake = np.clip(flake, 0, max_layers)
        label = np.maximum(label, flake)          # overlapping flakes -> thicker

    # Colour each pixel by its layer count.
    img = colors[label]                            # HxWx3

    # Realistic nuisances.
    illum = _illumination(rng, H, W)[..., None]
    img = img * illum
    # per-channel colour/white-balance jitter
    img *= rng.uniform(0.92, 1.08, size=(1, 1, 3)).astype(np.float32)
    img += rng.uniform(-0.03, 0.03, size=(1, 1, 3)).astype(np.float32)
    # sensor noise
    img += rng.normal(0, rng.uniform(0.005, 0.025), size=img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)

    # defocus blur (keeps labels sharp, blurs image only)
    pil = Image.fromarray((img * 255).astype(np.uint8))
    pil = pil.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.2)))
    img = np.asarray(pil, np.float32) / 255.0

    return img, label


class SyntheticFlakes(Dataset):
    """On-the-fly physics-based synthetic dataset."""

    # disjoint seed offsets guarantee no scene is shared across splits
    _SPLIT_OFFSET = {"train": 0, "val": 10_000_000, "test": 20_000_000}

    def __init__(self, n_samples, size=128, max_layers=4,
                 material="graphene", sio2_nm=285.0, split="train"):
        self.n = n_samples
        self.size = size
        self.max_layers = max_layers
        self.colors = color_table(max_layers, material, sio2_nm).astype(np.float32)
        self.offset = self._SPLIT_OFFSET[split]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.offset + idx)   # deterministic scene
        img, label = make_scene(rng, self.size, self.max_layers, self.colors)
        img = normalize_img(img)
        return to_chw_tensor(img), torch.from_numpy(label)


class RealFlakes(Dataset):
    """
    Real annotated data.

    Expects:
        root/images/<name>.(png|jpg)
        root/masks/<name>.png     # single channel, pixel value = layer count
    """

    def __init__(self, root, size=128, max_layers=4):
        from pathlib import Path
        self.root = Path(root)
        self.size = size
        self.max_layers = max_layers
        img_dir = self.root / "images"
        self.items = sorted(p for p in img_dir.iterdir()
                            if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if not self.items:
            raise FileNotFoundError(f"No images found in {img_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        from pathlib import Path
        ip = self.items[idx]
        mp = self.root / "masks" / (ip.stem + ".png")
        img = Image.open(ip).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        msk = Image.open(mp).resize((self.size, self.size), Image.NEAREST)
        img = np.asarray(img, np.float32) / 255.0
        label = np.clip(np.asarray(msk, np.int64), 0, self.max_layers)
        img = normalize_img(img)
        return to_chw_tensor(img), torch.from_numpy(label)


if __name__ == "__main__":
    # Save a small grid of synthetic samples for visual inspection.
    ds = SyntheticFlakes(n_samples=6, size=128, max_layers=4)
    tiles = []
    for i in range(6):
        x, y = ds[i]
        rgb = ((x.numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255).astype(np.uint8)
        tiles.append(rgb)
    grid = np.concatenate([np.concatenate(tiles[:3], 1),
                           np.concatenate(tiles[3:], 1)], 0)
    Image.fromarray(grid).save("outputs/sample_scenes.png")
    print("wrote outputs/sample_scenes.png  |  label values seen:",
          np.unique(ds[0][1].numpy()))
