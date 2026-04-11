"""
Macenko (2009) H&E stain normalization.

Normalizes histopathology patches to match the stain appearance of a reference image,
reducing the color distribution shift across hospital centers.

Usage as extra_transform (recommended — applied on raw uint8 before imagenet normalize):

    from utils.stain_normalization import MacenkoNormalizer, fit_normalizer_from_h5
    from utils.data_augmentation import Compose

    normalizer = fit_normalizer_from_h5("train.h5", n_reference_patches=200)
    extra_transform = Compose((normalizer,), allows_embedding_precompute=True)

    ProcessingConfig(
        resize_hw=(224, 224),
        imagenet_normalize=True,
        extra_transform=extra_transform,
    )

References:
    Macenko et al. (2009) "A method for normalizing histology slides for quantitative analysis"
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class MacenkoNormalizer:
    """Macenko stain normalizer — sklearn-compatible and callable as extra_transform.

    Accepts (C, H, W) numpy arrays (uint8 [0,255] or float [0,1]) and returns the same
    format after normalizing the H&E stain to match a fitted reference image.

    This transform is **deterministic** (no randomness), so it sets
    ``allows_embedding_precompute = True``, enabling frozen backbone features to be
    cached once when used in ``ProcessingConfig.extra_transform``.

    Picklable: stores only plain numpy arrays after fitting.
    """

    allows_embedding_precompute: bool = True

    def __init__(
        self,
        luminosity_threshold: float = 0.8,
        angular_percentile: float = 99.0,
    ) -> None:
        """
        Args:
            luminosity_threshold: Pixels with mean intensity > this * 255 are treated as
                background and excluded from stain estimation. Default 0.8.
            angular_percentile: Percentile used to identify the two extreme stain directions
                in the optical density plane. Default 99.
        """
        self.luminosity_threshold = float(luminosity_threshold)
        self.angular_percentile = float(angular_percentile)

        # Set after fit():
        self.stain_matrix_target_: Optional[np.ndarray] = None   # (2, 3) float64
        self.max_conc_target_: Optional[np.ndarray] = None        # (2,)  float64

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_rgb_hwc_uint8(x: np.ndarray) -> np.ndarray:
        """Convert any (C,H,W) or (H,W,C) array to (H,W,3) uint8."""
        x = np.asarray(x)
        # Handle CHW → HWC
        if x.ndim == 3 and x.shape[0] == 3:
            x = np.moveaxis(x, 0, -1)
        # Scale to uint8
        if x.dtype != np.uint8:
            if float(np.nanmax(x)) <= 1.5:
                x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
            else:
                x = np.clip(x, 0, 255).astype(np.uint8)
        return x  # (H, W, 3)

    @staticmethod
    def _original_scale(x_chw: np.ndarray) -> str:
        """Detect whether the CHW array is uint8 [0,255] or float [0,1]."""
        if x_chw.dtype == np.uint8 or float(np.nanmax(x_chw)) > 1.5:
            return "uint8"
        return "float01"

    @staticmethod
    def _rgb_to_od(rgb_hwc_u8: np.ndarray) -> np.ndarray:
        """Convert (H,W,3) uint8 → optical density (N,3), background mask applied."""
        rgb = rgb_hwc_u8.reshape(-1, 3).astype(np.float64)
        od = -np.log((rgb + 1.0) / 256.0)
        return od

    @classmethod
    def _extract_stain_matrix(
        cls,
        rgb_hwc_u8: np.ndarray,
        luminosity_threshold: float,
        angular_percentile: float,
    ) -> np.ndarray:
        """Extract the (2, 3) stain matrix from an RGB image via SVD + angular extremes."""
        # --- background mask (high luminosity = white/empty tissue) ---
        mean_intensity = rgb_hwc_u8.mean(axis=-1)  # (H, W)
        fg_mask = (mean_intensity < luminosity_threshold * 255).reshape(-1)
        od_all = cls._rgb_to_od(rgb_hwc_u8)        # (N, 3)
        od_fg = od_all[fg_mask]

        # Fallback: if too few foreground pixels use all
        if len(od_fg) < 20:
            od_fg = od_all

        # --- filter near-zero OD (transparent / background) ---
        od_norm = np.linalg.norm(od_fg, axis=1)
        od_fg = od_fg[od_norm > 0.15]

        if len(od_fg) < 20:
            # Degenerate patch (nearly white/black) — return canonical H&E matrix
            return np.array(
                [[0.6500, 0.7000, 0.2900],
                 [0.0700, 0.9900, 0.1100]],
                dtype=np.float64,
            )

        # --- SVD to find the 2D plane spanned by the stains ---
        _, _, Vt = np.linalg.svd(od_fg, full_matrices=False)
        plane = Vt[:2]  # (2, 3) — top 2 right singular vectors

        # --- project OD onto the plane and find angular extremes ---
        proj = od_fg @ plane.T  # (N, 2)
        angles = np.arctan2(proj[:, 1], proj[:, 0])
        a_low = np.percentile(angles, 100.0 - angular_percentile)
        a_high = np.percentile(angles, angular_percentile)

        # Back-project to 3D OD
        s1 = np.array([np.cos(a_low),  np.sin(a_low)])  @ plane  # (3,)
        s2 = np.array([np.cos(a_high), np.sin(a_high)]) @ plane  # (3,)

        # Ensure each stain vector has positive OD (absorbs light)
        if s1.sum() < 0:
            s1 = -s1
        if s2.sum() < 0:
            s2 = -s2

        stain_mat = np.stack([s1, s2], axis=0)  # (2, 3)

        # Row-normalize
        norms = np.linalg.norm(stain_mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return (stain_mat / norms).astype(np.float64)

    @staticmethod
    def _get_concentrations(od_flat: np.ndarray, stain_mat: np.ndarray) -> np.ndarray:
        """Solve OD ≈ C @ stain_mat via least squares. Returns (N, 2)."""
        # lstsq: minimize ||stain_mat.T @ c - od.T||  per pixel
        conc, _, _, _ = np.linalg.lstsq(stain_mat.T, od_flat.T, rcond=None)
        return conc.T  # (N, 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, reference: np.ndarray) -> "MacenkoNormalizer":
        """Fit to a reference image.

        Args:
            reference: (C, H, W) or (H, W, C) numpy array — the target appearance.
                Can be uint8 [0,255] or float [0,1].

        Returns:
            self (for chaining)
        """
        rgb_hwc = self._to_rgb_hwc_uint8(reference)
        self.stain_matrix_target_ = self._extract_stain_matrix(
            rgb_hwc, self.luminosity_threshold, self.angular_percentile
        )
        od = self._rgb_to_od(rgb_hwc)
        conc = self._get_concentrations(od, self.stain_matrix_target_)
        self.max_conc_target_ = np.percentile(conc, 99.0, axis=0).astype(np.float64)
        return self

    def transform(self, x_chw: np.ndarray) -> np.ndarray:
        """Normalize a patch to the reference stain.

        Args:
            x_chw: (C, H, W) numpy array — uint8 [0,255] or float [0,1].

        Returns:
            Normalized (C, H, W) array in the same dtype/scale as the input.
        """
        if self.stain_matrix_target_ is None or self.max_conc_target_ is None:
            raise RuntimeError("Call fit() before transform().")

        scale = self._original_scale(x_chw)
        rgb_hwc = self._to_rgb_hwc_uint8(x_chw)
        h, w = rgb_hwc.shape[:2]

        # Source stain matrix & concentrations
        src_mat = self._extract_stain_matrix(
            rgb_hwc, self.luminosity_threshold, self.angular_percentile
        )
        od = self._rgb_to_od(rgb_hwc)                           # (H*W, 3)
        src_conc = self._get_concentrations(od, src_mat)        # (H*W, 2)

        # Rescale concentrations to target range
        max_src = np.percentile(src_conc, 99.0, axis=0)
        max_src = np.maximum(max_src, 1e-8)
        normalized_conc = src_conc / max_src * self.max_conc_target_  # (H*W, 2)

        # Reconstruct RGB using target stain matrix
        od_out = normalized_conc @ self.stain_matrix_target_    # (H*W, 3)
        rgb_out = 256.0 * np.exp(-od_out) - 1.0
        rgb_out = np.clip(rgb_out, 0, 255).astype(np.uint8)
        rgb_out_hwc = rgb_out.reshape(h, w, 3)

        # Back to (C, H, W)
        out_chw = np.moveaxis(rgb_out_hwc, -1, 0)               # (3, H, W)
        if scale == "float01":
            return out_chw.astype(np.float32) / 255.0
        return out_chw.astype(np.uint8)

    def __call__(self, x_chw: np.ndarray) -> np.ndarray:
        """Alias for transform — makes this usable as extra_transform directly."""
        return self.transform(x_chw)


# ------------------------------------------------------------------
# Convenience: fit from a sample of train.h5 patches
# ------------------------------------------------------------------

def fit_normalizer_from_h5(
    h5_path: str,
    n_reference_patches: int = 200,
    target_center: Optional[int] = None,
    luminosity_threshold: float = 0.8,
    angular_percentile: float = 99.0,
    seed: int = 42,
) -> MacenkoNormalizer:
    """Fit a MacenkoNormalizer on a sample of patches from an HDF5 archive.

    Aggregates a representative "average" reference stain by stacking pixels from
    multiple patches and fitting a single normalizer to the combined image.

    Args:
        h5_path: Path to the HDF5 file (e.g. "train.h5").
        n_reference_patches: How many patches to sample for the reference.
        target_center: If given, only use patches from this center_id.
        luminosity_threshold: Passed to MacenkoNormalizer.
        angular_percentile: Passed to MacenkoNormalizer.
        seed: RNG seed for reproducible sampling.

    Returns:
        Fitted MacenkoNormalizer ready to use as extra_transform.
    """
    import h5py

    rng = np.random.default_rng(seed)

    with h5py.File(h5_path, "r") as f:
        all_keys = list(f.keys())

        if target_center is not None:
            filtered = []
            for k in all_keys:
                g = f[k]
                if "metadata" in g:
                    meta = np.array(g["metadata"]).reshape(-1)
                    if meta.size and int(meta[0]) == target_center:
                        filtered.append(k)
            candidate_keys = filtered if filtered else all_keys
        else:
            candidate_keys = all_keys

        n = min(n_reference_patches, len(candidate_keys))
        chosen = rng.choice(len(candidate_keys), size=n, replace=False)
        chosen_keys = [candidate_keys[i] for i in chosen]

        # Stack sampled patches as a single wide "reference image" (H, n*W, 3)
        patches = []
        for k in chosen_keys:
            img = np.array(f[k]["img"])               # (3, H, W) or (H, W, 3)
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.moveaxis(img, 0, -1)         # → (H, W, 3)
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            patches.append(img)

    # Concatenate along width axis for a single reference image
    reference = np.concatenate(patches, axis=1)       # (H, n*W, 3)

    normalizer = MacenkoNormalizer(
        luminosity_threshold=luminosity_threshold,
        angular_percentile=angular_percentile,
    )
    normalizer.fit(reference)
    return normalizer
