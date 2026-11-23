"""
dataset.py
==========
Turns the per-layer physics from `optics.py` into training data.

Two datasets are provided:

1. SyntheticFlakes -- generates labelled microscope-style scenes on the fly using
   the Fresnel colour model plus realistic nuisances (illumination gradients,
   vignetting, sensor noise, defocus blur, colour jitter). Each pixel's label is
   the integer number of layers (0 = bare substrate). Deterministic per index and
   split so train/val/test never share a scene (leakage-safe).

2. RealFlakes -- drop-in loader for real data: an images/ folder of RGB pictures
   and a masks/ folder of single-channel PNGs whose pixel values ARE the layer
   count (0,1,2,...). Use this once you have annotated microscope images.

Flake terraces are produced as level sets of a convex-polygon field, which yields
the straight-edged, angular, nested step structure that real exfoliated flakes
show (they cleave along crystal directions, so edges meet at ~60/120 deg).
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
    """Smooth field in [0,1] from a sum of rotated 2D Gaussians.

    Used for the low-frequency illumination field (which really is smooth).
    """
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


def _polygon_field(rng, H, W):
    """Angular field in [0,1] shaped like a real exfoliated flake.

    Builds a random convex polygon whose edges point along crystallographic
    directions (normals spaced 60 deg apart, optionally also the 30-deg-rotated
    set), each pushed a random distance from the centre so the polygon is
    irregular but straight-edged with sharp corners. The returned field is the
    inset distance from the edges: high in the middle, zero on the boundary.

    Thresholding this field at several levels (as make_scene does) yields nested,
    straight-edged terraces -- the stepped, angular look real flakes have, instead
    of smooth blobs.
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy = rng.uniform(0.32, 0.68) * H
    cx = rng.uniform(0.32, 0.68) * W
    th0 = rng.uniform(0.0, np.pi / 3.0)                 # random orientation (60 deg periodic)
    size = rng.uniform(0.18, 0.34) * min(H, W)          # overall flake radius

    # Edge normals: a hexagonal set (60 deg apart); half the time also add the
    # 30-deg-rotated set, giving up-to-12-sided straight-edged flakes.
    angles = [th0 + k * (np.pi / 3.0) for k in range(6)]
    if rng.random() < 0.5:
        angles += [th0 + np.pi / 6.0 + k * (np.pi / 3.0) for k in range(6)]

    field = np.full((H, W), 1e9, np.float32)
    for a in angles:
        ux, uy = np.cos(a), np.sin(a)
        r = size * rng.uniform(0.55, 1.0)               # this edge's distance from centre
        d = r - ((xx - cx) * ux + (yy - cy) * uy)       # > 0 inside this half-plane
        field = np.minimum(field, d)                    # inside ALL edges = the polygon

    field = np.clip(field, 0.0, None)
    field /= field.max() + 1e-9
    return field


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

    # 1..3 flakes, each a stack of terraces via level sets of a polygon field.
    for _ in range(rng.integers(1, 4)):
        g = _polygon_field(rng, H, W)                  # straight-edged, angular flake
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
