"""
train.py
========
Train the U-Net to predict per-pixel layer count (thickness) from flake images.

Data source:
  --data synthetic   (default) physics-based synthetic scenes, no files needed
  --data <path>      a RealFlakes root with images/ and masks/ subfolders

Example
-------
  python src/train.py --epochs 15 --train-size 2000 --material graphene
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import SyntheticFlakes, RealFlakes
from model import UNet
from metrics import confusion_matrix, summarise


def build_loaders(args):
    if args.data == "synthetic":
        train = SyntheticFlakes(args.train_size, args.size, args.max_layers,
                                args.material, split="train")
        val = SyntheticFlakes(max(args.train_size // 5, 64), args.size,
                              args.max_layers, args.material, split="val")
        test = SyntheticFlakes(max(args.train_size // 5, 64), args.size,
                               args.max_layers, args.material, split="test")
    else:
        full = RealFlakes(args.data, args.size, args.max_layers)
        n = len(full)
        n_val = max(1, int(0.15 * n)); n_test = max(1, int(0.15 * n))
        n_train = n - n_val - n_test
        train, val, test = random_split(
            full, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(0))

    mk = lambda ds, sh: DataLoader(ds, batch_size=args.batch, shuffle=sh,
                                   num_workers=args.workers, drop_last=False)
    return mk(train, True), mk(val, False), mk(test, False)


def class_weights(loader, n_classes, max_batches=40):
    """Inverse-frequency weights so thin, rare terraces are not ignored."""
    counts = np.zeros(n_classes, np.float64)
    for i, (_, y) in enumerate(loader):
        counts += np.bincount(y.reshape(-1).numpy(), minlength=n_classes)
        if i + 1 >= max_batches:
            break
    freq = counts / counts.sum()
    w = 1.0 / np.sqrt(freq + 1e-6)          # soft inverse-frequency
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate(net, loader, n_classes, device):
    net.eval()
    cm = np.zeros((n_classes, n_classes), np.int64)
    for x, y in loader:
        pred = net(x.to(device)).argmax(1).cpu()
        cm += confusion_matrix(pred, y, n_classes)
    return summarise(cm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synthetic")
    ap.add_argument("--material", default="graphene")
    ap.add_argument("--max-layers", type=int, default=4)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--train-size", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="checkpoints/best.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = args.max_layers + 1
    print(f"device={device}  classes={n_classes}  data={args.data}")

    train_loader, val_loader, test_loader = build_loaders(args)
    net = UNet(n_classes=n_classes).to(device)
    w = class_weights(train_loader, n_classes).to(device)
    print("class weights:", [round(float(x), 2) for x in w])

    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for ep in range(1, args.epochs + 1):
        net.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(net(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        sched.step()
        train_loss = running / len(train_loader.dataset)
        m = evaluate(net, val_loader, n_classes, device)
        flag = ""
        if m["mean_abs_layer_error"] < best:
            best = m["mean_abs_layer_error"]
            torch.save({"model": net.state_dict(),
                        "max_layers": args.max_layers,
                        "material": args.material}, args.out)
            flag = "  <- best (saved)"
        print(f"epoch {ep:02d}  loss={train_loss:.3f}  "
              f"val_acc={m['pixel_acc']:.3f}  val_mIoU={m['mean_iou']:.3f}  "
              f"val_layerMAE={m['mean_abs_layer_error']:.3f}{flag}")

    # Final test-set report using the best checkpoint.
    ckpt = torch.load(args.out, map_location=device)
    net.load_state_dict(ckpt["model"])
    t = evaluate(net, test_loader, n_classes, device)
    print("\n=== TEST ===")
    print(f"pixel_acc={t['pixel_acc']:.3f}  mIoU={t['mean_iou']:.3f}  "
          f"layerMAE={t['mean_abs_layer_error']:.3f}")
    print("per-class IoU:", [round(x, 3) for x in t['per_class_iou']])


if __name__ == "__main__":
    main()
