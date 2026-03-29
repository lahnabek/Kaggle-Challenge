"""
Histopathology OOD pipeline utilities (DINOv2, optional adapter, Method C outliers).

Import from submodules (e.g. ``from utils.config import RunConfig``) or use this package
as the single entry point after adding the repo root to ``sys.path`` (notebooks, Colab).
"""

from utils.config import (
    EarlyStoppingConfig,
    ModelConfig,
    ModuleSpec,
    ProcessingConfig,
    RunConfig,
    TrainConfig,
    save_run_config,
)
from utils.constants import (
    DEVICE,
    RUNS_DIR,
    TEST_IMAGES_PATH,
    TRAIN_IMAGES_PATH,
    VAL_IMAGES_PATH,
)
from utils.data import (
    H5BinaryDataset,
    PreprocessingTransform,
    build_preprocessing,
)
from utils.experiment import run_experiment
from utils.model import DefaultBinaryHead, DefaultMLPAdapter, FullModel
from utils.outliers import ExtractOutlier, MethodCOutlierParams

__all__ = [
    "DEVICE",
    "RUNS_DIR",
    "TEST_IMAGES_PATH",
    "TRAIN_IMAGES_PATH",
    "VAL_IMAGES_PATH",
    "EarlyStoppingConfig",
    "ModelConfig",
    "ModuleSpec",
    "ProcessingConfig",
    "RunConfig",
    "TrainConfig",
    "save_run_config",
    "H5BinaryDataset",
    "PreprocessingTransform",
    "build_preprocessing",
    "run_experiment",
    "DefaultBinaryHead",
    "DefaultMLPAdapter",
    "FullModel",
    "ExtractOutlier",
    "MethodCOutlierParams",
]
