"""
metrics.py

========


Evaluation metrics for layer-count segmentation.

- pixel_accuracy : overall fraction of correctly classified pixels
- confusion_matrix : (C x C) counts, rows = true, cols = pred
- per_class_iou / mean_iou : segmentation quality per layer class
- mean_abs_layer_error : average |pred_layers - true_layers| over FLAKE pixels
      (the physically meaningful number: how many layers off, on average)
"""

from __future__ import annotations
import numpy as np
import torch


@torch.no_grad()
def confusion_matrix(pred, target, n_classes):
    """pred, target: LongTensor of layer indices. Returns np CxC array."""
    p = pred.reshape(-1)
    t = target.reshape(-1)
    k = (t * n_classes + p).cpu().numpy()
    cm = np.bincount(k, minlength=n_classes * n_classes)
    return cm.reshape(n_classes, n_classes)


def summarise(cm):
    """Turn an accumulated confusion matrix into a metrics dict."""
    cm = cm.astype(np.float64)
    total = cm.sum()
    correct = np.trace(cm)
    pixel_acc = correct / max(total, 1)

    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    mean_iou = np.nanmean(iou)

    # mean absolute layer error over pixels whose TRUE label is a flake (>0)
    idx = np.arange(cm.shape[0])
    diff = np.abs(idx[:, None] - idx[None, :])          # |true - pred|
    flake_rows = cm[1:, :]                               # true layer >= 1
    n_flake = flake_rows.sum()
    male = (flake_rows * diff[1:, :]).sum() / max(n_flake, 1)

    return {
        "pixel_acc": float(pixel_acc),
        "mean_iou": float(mean_iou),
        "mean_abs_layer_error": float(male),
        "per_class_iou": [float(x) for x in iou],
    }
