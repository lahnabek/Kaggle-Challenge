"""
HDF5 datasets, embedding tensor datasets, and picklable preprocessing.

Classes defined in this module are importable as ``utils.data.*``, so ``DataLoader``
with ``num_workers > 0`` works under multiprocessing ``spawn`` (macOS, Windows, Colab).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Sampler

from utils.config import ProcessingConfig


class H5BinaryDataset(Dataset):
    """H5 dataset for tumor / no-tumor patch classification.

    Expected group per key: ``img`` (C,H,W), ``label`` (train/val), ``metadata`` (center id at [0]).
    """

    def __init__(
        self,
        h5_path: str,
        transform: Callable[[torch.Tensor], torch.Tensor],
        mode: str,
        outlier_ids: Optional[Set[str]] = None,
        subset_ids: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.h5_path = h5_path
        self.transform = transform
        self.mode = mode
        self.outlier_ids = outlier_ids

        self._file = None
        with h5py.File(self.h5_path, "r") as f:
            if subset_ids is not None:
                self.image_ids = list(subset_ids)
            else:
                self.image_ids = list(f.keys())

    def __getstate__(self) -> Dict[str, Any]:
        """Do not pickle an open HDF5 handle (each worker reopens the file)."""
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._file = None

    def _get_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        key = self.image_ids[idx]
        f = self._get_file()
        g = f[key]

        img = torch.tensor(np.array(g["img"]))
        x = self.transform(img)

        if self.mode in ("train", "val"):
            y = torch.tensor(np.array(g["label"]).reshape(-1)[0]).float()
            meta = np.array(g.get("metadata")) if "metadata" in g else np.array([np.nan])
            center_id = int(meta.reshape(-1)[0]) if meta.size else -1
            if self.outlier_ids is not None:
                is_out = str(key) in self.outlier_ids
                return x, y, center_id, torch.tensor(is_out, dtype=torch.bool)
            return x, y, center_id

        if self.outlier_ids is not None:
            is_out = str(key) in self.outlier_ids
            return x, int(key), torch.tensor(is_out, dtype=torch.bool)
        return x, int(key)


class EmbeddingTensorDataset(Dataset):
    """Precomputed backbone features for training: (z, y, center_id)."""

    def __init__(self, z: torch.Tensor, y: torch.Tensor, centers: torch.Tensor) -> None:
        super().__init__()
        self.z = z
        self.y = y
        self.centers = centers

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        return self.z[idx], self.y[idx], self.centers[idx]


class EmbeddingValDataset(Dataset):
    """Precomputed features for validation with optional outlier mask."""

    def __init__(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        centers: torch.Tensor,
        is_out: torch.Tensor,
    ) -> None:
        super().__init__()
        self.z = z
        self.y = y
        self.centers = centers
        self.is_out = is_out

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        return self.z[idx], self.y[idx], self.centers[idx], self.is_out[idx]


class SklearnLikeTransform:
    """Wrap a sklearn-like transformer (``transform`` or callable) for numpy (C,H,W) arrays."""

    def __init__(self, transformer: Any) -> None:
        self.transformer = transformer

    def __call__(self, x_np: np.ndarray) -> np.ndarray:
        if hasattr(self.transformer, "transform"):
            return self.transformer.transform(x_np)
        if callable(self.transformer):
            return self.transformer(x_np)
        raise TypeError("sklearn_transformer must be callable or implement .transform(x).")


class PreprocessingTransform:
    """Picklable preprocessing pipeline (required for ``DataLoader`` with ``num_workers > 0``).

    Must be a top-level class so worker processes can unpickle it; nested closures are not safe
    under ``spawn``. If you set ``extra_transform`` / ``sklearn_transformer`` to non-picklable
    objects, workers may still fail.
    """

    def __init__(self, cfg: ProcessingConfig) -> None:
        self.cfg = cfg
        self._resize = transforms.Resize(cfg.resize_hw)
        self._norm = (
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            if bool(cfg.imagenet_normalize)
            else None
        )
        self._sklearn = (
            SklearnLikeTransform(cfg.sklearn_transformer) if cfg.sklearn_transformer is not None else None
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = self._resize(x)
        if self.cfg.cast_float32:
            y = y.float()
        if self._norm is not None:
            y = self._norm(y)

        if self.cfg.extra_transform is not None:
            y_np = y.detach().cpu().numpy().astype(np.float32)
            y_np = self.cfg.extra_transform(y_np)
            y = torch.tensor(y_np, dtype=torch.float32)

        if self._sklearn is not None:
            y_np = y.detach().cpu().numpy().astype(np.float32)
            y_np = self._sklearn(y_np)
            y = torch.tensor(y_np, dtype=torch.float32)

        return y


def build_preprocessing(cfg: ProcessingConfig) -> PreprocessingTransform:
    """Build a picklable transform from ``ProcessingConfig``."""
    return PreprocessingTransform(cfg)


class CenterProportionalBatchSampler(Sampler[List[int]]):
    """Yields index lists per batch with approximate fixed center proportions (training only)."""

    def __init__(
        self,
        centers: List[int],
        batch_size: int,
        proportions: Dict[int, float],
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ) -> None:
        self.centers = np.array(centers, dtype=int)
        self.batch_size = int(batch_size)
        self.proportions = dict(proportions)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

        self.center_to_indices: Dict[int, Any] = {}
        for i, c in enumerate(self.centers.tolist()):
            self.center_to_indices.setdefault(int(c), []).append(i)

        for c in self.center_to_indices:
            self.center_to_indices[c] = np.array(self.center_to_indices[c], dtype=int)

        probs = {int(k): float(v) for k, v in self.proportions.items()}
        probs = {c: probs.get(c, 0.0) for c in self.center_to_indices.keys()}
        s = sum(probs.values())
        if s <= 0:
            probs = {c: 1.0 / len(probs) for c in probs}
        else:
            probs = {c: v / s for c, v in probs.items()}

        raw = {c: probs[c] * self.batch_size for c in probs}
        base = {c: int(np.floor(raw[c])) for c in raw}
        used = sum(base.values())
        remainder = self.batch_size - used
        frac = sorted([(c, raw[c] - base[c]) for c in raw], key=lambda t: t[1], reverse=True)
        for c, _ in frac[:remainder]:
            base[c] += 1

        self.per_batch = base

        limiting = []
        for c, cnt in self.per_batch.items():
            if cnt == 0:
                continue
            limiting.append(len(self.center_to_indices[c]) // cnt)
        self.num_batches = min(limiting) if limiting else 0

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        streams = {}
        for c, idxs in self.center_to_indices.items():
            idxs = idxs.copy()
            if self.shuffle:
                self.rng.shuffle(idxs)
            streams[c] = idxs

        ptr = {c: 0 for c in streams}

        for _ in range(self.num_batches):
            batch = []
            for c, cnt in self.per_batch.items():
                if cnt <= 0:
                    continue
                start = ptr[c]
                end = start + cnt
                batch.extend(streams[c][start:end].tolist())
                ptr[c] = end

            if len(batch) != self.batch_size:
                if self.drop_last:
                    continue
            if self.shuffle:
                self.rng.shuffle(batch)
            yield batch
