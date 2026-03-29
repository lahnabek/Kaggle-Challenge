"""
Single place for **torch device** and **default data / output paths**.

Edit values here (or change the process working directory) instead of scattering
constants across the package. Imported by ``experiment``, ``predict``, ``model``, etc.
"""

from __future__ import annotations

import torch

__all__ = [
    "DEVICE",
    "TRAIN_IMAGES_PATH",
    "VAL_IMAGES_PATH",
    "TEST_IMAGES_PATH",
    "RUNS_DIR",
]

# Compute device for model / tensors (CUDA when available).
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HDF5 patch archives (paths relative to the current working directory).
TRAIN_IMAGES_PATH = "train.h5"
VAL_IMAGES_PATH = "val.h5"
TEST_IMAGES_PATH = "test.h5"

# Root directory for experiment outputs (metrics, checkpoints, predictions).
RUNS_DIR = "runs"
