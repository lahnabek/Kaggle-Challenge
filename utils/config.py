"""Experiment configuration dataclasses and JSON export."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from utils.common import ensure_dir, safe_json
from utils.outliers import MethodCOutlierParams


@dataclass
class ProcessingConfig:
    """Image preprocessing configuration.

    Order: resize (torchvision) → optional ``extra_transform`` on numpy (C,H,W) → optional sklearn-like step.
    """

    resize_hw: Tuple[int, int] = (98, 98)
    cast_float32: bool = True
    extra_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
    sklearn_transformer: Any = None


@dataclass
class ModuleSpec:
    """Generic module specification (adapter or head).

    The module class must accept ``in_dim`` as a keyword argument.
    """

    enabled: bool = True
    module_cls: Optional[type] = None
    module_kwargs: Optional[Dict[str, Any]] = None


@dataclass
class ModelConfig:
    """Model configuration.

    If ``adapter.enabled`` is False, training **precomputes** frozen backbone embeddings once
    and trains only the head on tensors (no repeated DINO forwards across epochs).
    """

    backbone_name: str = "dinov2_vits14"
    adapter: ModuleSpec = field(
        default_factory=lambda: ModuleSpec(enabled=True, module_cls=None, module_kwargs=None)
    )
    head: ModuleSpec = field(
        default_factory=lambda: ModuleSpec(enabled=True, module_cls=None, module_kwargs=None)
    )


@dataclass
class EarlyStoppingConfig:
    """Early stopping: which validation metric to monitor and patience."""

    monitor: str = "val_loss"
    mode: str = "min"
    patience: int = 10
    min_delta: float = 0.0


@dataclass
class TrainConfig:
    """Optimizer, loader, and early-stopping hyperparameters."""

    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 0.0
    num_epochs: int = 50

    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)

    use_center_balanced_batches: bool = False

    center_proportions: Optional[Dict[int, float]] = None

    num_workers: int = 8
    drop_last: bool = True


@dataclass
class RunConfig:
    """One full experiment: data, model, train, optional outlier filter and test prediction."""

    run_name: str
    seeds: List[int]
    processing: ProcessingConfig
    model: ModelConfig
    train: TrainConfig

    do_predict_test: bool = False
    predict_threshold: float = 0.5

    # Method C outlier filtering (``ExtractOutlier``); ``None`` = disabled.
    outlier_params: Optional[MethodCOutlierParams] = None
    outlier_scan_workers: int = 4
    outlier_scan_batch_size: int = 256

    # Random fraction of train/val/test IDs after outlier filter (smoke tests; ``None`` = full data).
    data_fraction: Optional[float] = None


def save_run_config(run_cfg: RunConfig, run_dir: str) -> None:
    """Save ``config.json`` under the run directory for reproducibility."""
    ensure_dir(run_dir)

    d = asdict(run_cfg)
    d["processing"]["sklearn_transformer"] = safe_json(run_cfg.processing.sklearn_transformer)
    d["model"]["adapter"]["module_cls"] = safe_json(run_cfg.model.adapter.module_cls)
    d["model"]["head"]["module_cls"] = safe_json(run_cfg.model.head.module_cls)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(d, f, indent=2)
