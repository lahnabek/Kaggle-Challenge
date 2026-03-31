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
class AugmentationConfig:
    """Multi-view data augmentation configuration (batch augmentation / repeated augmentation).

    If ``k_aug_views > 0``, ``ProcessingConfig.extra_transform`` is applied *k* times
    independently to produce k augmented views. If ``include_original`` is True, an
    un-augmented view is also included.

    The preprocessing output becomes (V, C, H, W) per sample where V = include_original + k_aug_views.
    """

    include_original: bool = True
    k_aug_views: int = 0


@dataclass
class ProcessingConfig:
    """Image preprocessing configuration.

    Order: resize (torchvision) → optional ``extra_transform`` on numpy (C,H,W) → optional sklearn-like step.
    """

    resize_hw: Tuple[int, int] = (98, 98)
    cast_float32: bool = True
    imagenet_normalize: bool = False
    extra_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
    sklearn_transformer: Any = None
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


def processing_allows_embedding_precompute(cfg: ProcessingConfig) -> bool:
    """Whether linear probing may precompute frozen backbone embeddings once.

    Returns False when:
    - multi-view training (``k_aug_views > 0``): views differ per epoch / sample.
    - ``extra_transform`` is set and does not declare ``allows_embedding_precompute``
      (custom augmentations default to online backbone).
    - ``extra_transform`` explicitly sets ``allows_embedding_precompute=False``
      (e.g. H&E RandAugment from ``make_train_augmentations``).

    Jitter-only pipelines from ``make_train_augmentations(use_he_randaugment=False)``
    set ``allows_embedding_precompute=True`` on the returned ``Compose``.
    """
    aug = getattr(cfg, "augmentation", None)
    if aug is not None and int(getattr(aug, "k_aug_views", 0) or 0) > 0:
        return False
    if cfg.extra_transform is None:
        return True
    return bool(getattr(cfg.extra_transform, "allows_embedding_precompute", False))


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

    If ``adapter.enabled`` is False, training normally **precomputes** frozen backbone
    embeddings once and trains only the head—except when preprocessing uses stochastic
    histology augmentations or multi-view sampling; see ``processing_allows_embedding_precompute``.
    """

    backbone_name: str = "dinov2_vits14"
    use_fp16: bool = False
    adapter: ModuleSpec = field(
        default_factory=lambda: ModuleSpec(enabled=True, module_cls=None, module_kwargs=None)
    )
    head: ModuleSpec = field(
        default_factory=lambda: ModuleSpec(enabled=True, module_cls=None, module_kwargs=None)
    )
    # Image augmentation is configured in `ProcessingConfig.augmentation`.


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

    # If `use_center_balanced_batches` is True and `center_proportions` is None:
    # - "empirical": use the observed train distribution
    # - "uniform": force equal per-center exposure (oversampling minor centers with replacement)
    center_sampling: str = "empirical"

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
