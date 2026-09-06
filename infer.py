"""
infer.py
========
Final file: runs a trained model on an image and produce a thickness map.

Usage
-----
  # on your own microscope image
  python src/infer.py --ckpt checkpoints/best.pt --image path/to/flake.jpg

  # no image handy? generate a physics-based synthetic one and analyse it
  python src/infer.py --ckpt checkpoints/best.pt --demo

Outputs (in outputs/):
  prediction.png   input | predicted thickness map | layer histogram
and prints a per-layer area / mean-thickness summary to stdout.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from model import UNet
from dataset import normalize_img, to_chw_tensor, SyntheticFlakes
from optics import MATERIALS


def load_image(path, size):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--image", default=None)
    ap.add_argument("--demo", action="store_true",
                    help="analyse a generated synthetic scene instead of a file")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--out", default="outputs/prediction.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    max_layers = ckpt["max_layers"]
    material = ckpt.get("material", "graphene")
    d_layer = MATERIALS[material]["d_layer"]

    net = UNet(n_classes=max_layers + 1).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()

    if args.demo or args.image is None:
        ds = SyntheticFlakes(1, args.size, max_layers, material, split="test")
        x, _ = ds[np.random.randint(0, 10_000)]
        rgb = (x.numpy().transpose(1, 2, 0) * 0.5 + 0.5)
    else:
        rgb = load_image(args.image, args.size)
        x = to_chw_tensor(normalize_img(rgb))

    with torch.no_grad():
        layers = net(x.unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()

    thickness_nm = layers * d_layer

    # ---- summary ----
    px = layers.size
    print(f"material={material}  d_layer={d_layer} nm  image={args.size}x{args.size}")
    print(f"{'layers':>6} {'thickness(nm)':>13} {'area(%)':>8}")
    for n in range(max_layers + 1):
        frac = 100.0 * (layers == n).mean()
        tag = "substrate" if n == 0 else ""
        print(f"{n:>6} {n*d_layer:>13.3f} {frac:>8.1f}  {tag}")
    flake = layers > 0
    if flake.any():
        print(f"flake coverage: {100*flake.mean():.1f}%   "
              f"mean flake thickness: {thickness_nm[flake].mean():.3f} nm   "
              f"max: {thickness_nm.max():.3f} nm")

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    ax[0].imshow(rgb); ax[0].set_title("input"); ax[0].axis("off")

    im = ax[1].imshow(thickness_nm, cmap="viridis",
                      vmin=0, vmax=max_layers * d_layer)
    ax[1].set_title("predicted thickness"); ax[1].axis("off")
    cb = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    cb.set_label("nm")

    counts = [np.mean(layers == n) * 100 for n in range(max_layers + 1)]
    ax[2].bar(range(max_layers + 1), counts, color="#4c72b0")
    ax[2].set_xlabel("layer count"); ax[2].set_ylabel("area (%)")
    ax[2].set_title("layer distribution")
    fig.tight_layout()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
