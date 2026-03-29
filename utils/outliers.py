"""
Method C outlier detection (color / texture / white-region rules).

Compatible with ``outlier_detection.ipynb``: same thresholds on raw H5 tensors
(native resolution, not the training resize). Scanning uses **threads** (ThreadPoolExecutor),
not PyTorch DataLoader workers — independent of training ``num_workers``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from tqdm.auto import tqdm


@dataclass
class MethodCOutlierParams:
    """Method C thresholds — same rules as ``outlier_detection.ipynb`` (``method_c_apply_thresholds`` + soft white).

    Match a typical grid row: ``min_sat_mean``, ``min_colorfulness``, ``min_grad_energy``, ``max_white_frac``;
    ``white_gray_min`` / ``white_max_sat`` are the shared H5-pass knobs (defaults match ``extract_method_c_features``).
    """

    min_sat_mean: float = 0.03
    min_colorfulness: float = 0.03
    min_grad_energy: float = 0.005
    white_gray_min: float = 0.86
    white_max_sat: float = 0.28
    max_white_frac: float = 0.8 # 0.6, 0.7, 0.8


def _to_hwc_float01(x: np.ndarray, patch_id: str) -> np.ndarray:
    """CHW→HWC, [0,1] float (same convention as outlier_detection)."""
    if x.ndim != 3:
        raise ValueError(f"Unexpected patch shape for ID {patch_id}: {x.shape}")
    if x.shape[0] in (1, 3):
        x = np.moveaxis(x, 0, -1)
    if x.shape[-1] == 1:
        x = np.repeat(x, repeats=3, axis=-1)
    x = np.asarray(x)
    if x.dtype == np.uint8 or float(np.nanmax(x)) > 1.5:
        x = x.astype(np.float32) / 255.0
    else:
        x = x.astype(np.float32)
    return np.clip(x, 0.0, 1.0)


def _rgb_to_gray(img: np.ndarray) -> np.ndarray:
    imgf = img.astype(np.float32)
    return 0.299 * imgf[..., 0] + 0.587 * imgf[..., 1] + 0.114 * imgf[..., 2]


def _hsv_saturation_mean(img: np.ndarray) -> float:
    x = img.astype(np.float32)
    mx = np.max(x, axis=-1)
    mn = np.min(x, axis=-1)
    sat = np.where(mx > 0, (mx - mn) / (mx + 1e-8), 0.0)
    return float(np.mean(sat))


def _rgb_local_saturation_map(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    mx = np.max(x, axis=-1)
    mn = np.min(x, axis=-1)
    return np.where(mx > 0, (mx - mn) / (mx + 1e-8), 0.0)


def _colorfulness(img: np.ndarray) -> float:
    r = img[..., 0].astype(np.float32)
    g = img[..., 1].astype(np.float32)
    b = img[..., 2].astype(np.float32)
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std_rg = np.std(rg)
    std_yb = np.std(yb)
    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)
    return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))


def _gradient_energy(gray: np.ndarray) -> float:
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    return float((gx.mean() + gy.mean()) / 2.0)


def _p_soft_white(img: np.ndarray, white_gray_min: float, white_max_sat: float) -> float:
    gray = _rgb_to_gray(img)
    gmin = float(white_gray_min)
    smax = float(white_max_sat)
    sat_px = _rgb_local_saturation_map(img)
    if smax >= 1.0 - 1e-9:
        soft_mask = gray >= gmin
    else:
        soft_mask = (gray >= gmin) & (sat_px <= smax)
    return float(np.mean(soft_mask))


def method_c_is_outlier(img: np.ndarray, p: MethodCOutlierParams) -> bool:
    """Return True if patch is flagged outlier (same boolean structure as ``method_c_apply_thresholds``)."""
    gray = _rgb_to_gray(img)
    sat = _hsv_saturation_mean(img)
    cful = _colorfulness(img)
    grad = _gradient_energy(gray)
    pw = _p_soft_white(img, p.white_gray_min, p.white_max_sat)
    base_out = ((sat < p.min_sat_mean) and (cful < p.min_colorfulness)) or (grad < p.min_grad_energy)
    return bool(base_out and (pw <= float(p.max_white_frac)))


def _scan_chunk(h5_path: str, patch_ids: List[str], p: MethodCOutlierParams) -> List[Tuple[str, bool]]:
    out: List[Tuple[str, bool]] = []
    with h5py.File(h5_path, "r") as f:
        for k in patch_ids:
            img = _to_hwc_float01(np.array(f[k]["img"]), str(k))
            out.append((str(k), method_c_is_outlier(img, p)))
    return out


class ExtractOutlier:
    """Parallel scan of H5 patches with method C outlier rules (thread pool over chunk tasks)."""

    def __init__(self, params: MethodCOutlierParams) -> None:
        self.params = params

    def scan(
        self,
        h5_path: str,
        patch_ids: List[str],
        *,
        batch_size: int = 256,
        num_workers: int = 4,
        desc: str = "outlier_scan",
    ) -> Dict[str, bool]:
        """Return ``patch_id -> is_outlier`` for all IDs in ``patch_ids``."""
        ids = list(patch_ids)
        if not ids:
            return {}
        w = max(1, int(num_workers))
        bs = max(1, int(batch_size))
        bounds = np.linspace(0, len(ids), w + 1, dtype=int)
        partitions = [ids[bounds[i] : bounds[i + 1]] for i in range(w) if bounds[i] < bounds[i + 1]]
        tasks: List[List[str]] = []
        for part in partitions:
            for j in range(0, len(part), bs):
                tasks.append(part[j : j + bs])

        worker = partial(_scan_chunk, h5_path, p=self.params)

        if w == 1:
            merged: List[Tuple[str, bool]] = []
            for t in tqdm(tasks, desc=desc, leave=False):
                merged.extend(worker(t))
            return dict(merged)

        rows_by_i: List[Optional[List[Tuple[str, bool]]]] = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=w) as ex:
            futs = {ex.submit(worker, t): idx for idx, t in enumerate(tasks)}
            for fut in tqdm(as_completed(futs), total=len(futs), desc=desc, leave=False):
                idx = futs[fut]
                rows_by_i[idx] = fut.result()

        merged = []
        for part in rows_by_i:
            if part:
                merged.extend(part)
        return dict(merged)
