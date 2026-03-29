"""Small helpers: seeding, filesystem, JSON-safe config serialization."""

from __future__ import annotations

import json
import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def safe_json(x: Any) -> Any:
    """Best-effort JSON serialization for config objects (classes become names or str)."""
    try:
        json.dumps(x)
        return x
    except TypeError:
        if hasattr(x, "__name__"):
            return x.__name__
        return str(x)
