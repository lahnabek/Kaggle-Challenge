"""
Histopathology-focused data augmentation.

Main target: fast, stain-aware augmentation inspired by
Faryna et al. (MIDL 2021) "Tailoring automated data augmentation to H&E-stained histopathology"
and the reference implementation vendored under `pathology-he-auto-augment/`.

All transforms here are **picklable** (top-level classes) so they can be used inside
`DataLoader(num_workers>0)` on macOS (spawn).

Expected array convention for callables:
- numpy array shaped (C, H, W)
- dtype uint8 in [0,255] OR float in [0,1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from PIL import Image, ImageEnhance, ImageOps
except Exception as e:  # pragma: no cover
    raise ImportError("PIL is required for data augmentation. Install pillow.") from e


def _chw_to_hwc(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Expected CHW array, got shape {x.shape}")
    if x.shape[0] not in (1, 3):
        raise ValueError(f"Expected C=1 or 3 in CHW, got shape {x.shape}")
    return np.moveaxis(x, 0, -1)


def _hwc_to_chw(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Expected HWC array, got shape {x.shape}")
    if x.shape[-1] not in (1, 3):
        raise ValueError(f"Expected last dim 1 or 3 in HWC, got shape {x.shape}")
    return np.moveaxis(x, -1, 0)


def _to_uint8_hwc(x_chw: np.ndarray) -> Tuple[np.ndarray, str]:
    """Return HWC uint8 image + a tag describing original scale."""
    x = np.asarray(x_chw)
    scale = "uint8"
    if x.dtype == np.uint8 or float(np.nanmax(x)) > 1.5:
        x_u8 = x.astype(np.uint8)
    else:
        # assume float in [0,1]
        scale = "float01"
        x_u8 = np.clip(x * 255.0, 0.0, 255.0).astype(np.uint8)
    x_hwc = _chw_to_hwc(x_u8)
    if x_hwc.shape[-1] == 1:
        x_hwc = np.repeat(x_hwc, repeats=3, axis=-1)
    return x_hwc, scale


def _from_uint8_hwc(x_hwc_u8: np.ndarray, scale: str) -> np.ndarray:
    x_hwc_u8 = np.asarray(x_hwc_u8, dtype=np.uint8)
    x_chw_u8 = _hwc_to_chw(x_hwc_u8)
    if scale == "uint8":
        return x_chw_u8
    if scale == "float01":
        return (x_chw_u8.astype(np.float32) / 255.0).astype(np.float32)
    raise ValueError(f"Unknown scale tag: {scale}")


# ---- Minimal HED conversion (copied from pathology-he-auto-augment/he-randaugment/custom_hed_transform.py) ----

def _rgb_from_hed() -> np.ndarray:
    return np.array(
        [
            [0.65, 0.70, 0.29],
            [0.07, 0.99, 0.11],
            [0.27, 0.57, 0.78],
        ],
        dtype=np.float32,
    )


_RGB_FROM_HED = _rgb_from_hed()
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED).astype(np.float32)


def rgb2hed(rgb_float01: np.ndarray) -> np.ndarray:
    rgb = rgb_float01.astype(np.float32, copy=True)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = rgb + 2.0
    stains = np.dot(np.reshape(-np.log(rgb), (-1, 3)), _HED_FROM_RGB)
    return np.reshape(stains, rgb.shape).astype(np.float32)


def hed2rgb(hed: np.ndarray) -> np.ndarray:
    stains = hed.astype(np.float32, copy=False)
    logrgb2 = np.dot(-np.reshape(stains, (-1, 3)), _RGB_FROM_HED)
    rgb2 = np.exp(logrgb2).reshape(stains.shape).astype(np.float32)
    rgb = rgb2 - 2.0
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def hed_shift_uint8(img_u8_hwc: np.ndarray, factor: float, rng: np.random.Generator) -> np.ndarray:
    """HED shift: random sigma/bias per H/E/D channel in [-factor, factor] (Faryna et al.)."""
    x = img_u8_hwc.astype(np.float32) / 255.0
    hed = rgb2hed(x)
    sigmas = rng.uniform(low=-factor, high=factor, size=(3,)).astype(np.float32)
    biases = rng.uniform(low=-factor, high=factor, size=(3,)).astype(np.float32)
    hed = hed * (1.0 + sigmas.reshape(1, 1, 3)) + biases.reshape(1, 1, 3)
    rgb = hed2rgb(hed)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def hsv_shift_uint8(img_u8_hwc: np.ndarray, factor: float, rng: np.random.Generator) -> np.ndarray:
    """HSV shift using skimage (hue/sat only, like repo's HsbColorAugmenter use-case)."""
    try:
        import skimage.color  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("HSV shift requires scikit-image (skimage). Install scikit-image.") from e

    x = img_u8_hwc.astype(np.float32) / 255.0
    hsv = skimage.color.rgb2hsv(x)
    dh = float(rng.uniform(-factor, factor))
    ds = float(rng.uniform(-factor, factor))

    hsv[..., 0] = (hsv[..., 0] + (dh % 1.0)) % 1.0
    if ds < 0:
        hsv[..., 1] = hsv[..., 1] * (1.0 + ds)
    else:
        hsv[..., 1] = hsv[..., 1] * (1.0 + (1.0 - hsv[..., 1]) * ds)
    hsv[..., 1] = np.clip(hsv[..., 1], 0.0, 1.0)

    rgb = skimage.color.hsv2rgb(hsv)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


# ---- Basic color jitter (cheap baseline) ----


@dataclass(frozen=True)
class BasicColorJitter:
    """Simple PIL-based jitter for brightness/contrast/color.

    Factors are sampled as:
      brightness ~ U(1-b, 1+b)
      contrast   ~ U(1-c, 1+c)
      color      ~ U(1-s, 1+s)
    """

    brightness: float = 0.1
    contrast: float = 0.1
    saturation: float = 0.1
    p: float = 1.0

    def __call__(self, x_chw: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        if float(rng.random()) > float(self.p):
            return x_chw

        img_u8_hwc, scale = _to_uint8_hwc(x_chw)
        pil = Image.fromarray(img_u8_hwc)

        ops: List[Callable[[Image.Image], Image.Image]] = []
        if self.brightness > 0:
            b = float(rng.uniform(1.0 - self.brightness, 1.0 + self.brightness))
            ops.append(lambda im, b=b: ImageEnhance.Brightness(im).enhance(b))
        if self.contrast > 0:
            c = float(rng.uniform(1.0 - self.contrast, 1.0 + self.contrast))
            ops.append(lambda im, c=c: ImageEnhance.Contrast(im).enhance(c))
        if self.saturation > 0:
            s = float(rng.uniform(1.0 - self.saturation, 1.0 + self.saturation))
            ops.append(lambda im, s=s: ImageEnhance.Color(im).enhance(s))

        rng.shuffle(ops)
        for op in ops:
            pil = op(pil)

        out = np.asarray(pil, dtype=np.uint8)
        return _from_uint8_hwc(out, scale=scale)


# ---- H&E tailored RandAugment (Faryna et al.) ----


_DEFAULT_OPS: Tuple[str, ...] = (
    "TranslateX",
    "TranslateY",
    "ShearX",
    "ShearY",
    "Brightness",
    "Sharpness",
    "Color",
    "Contrast",
    "Rotate",
    "Equalize",
    "Identity",
    "Hsv",
    "Hed",
)


def _maybe_neg(v: float, rng: np.random.Generator) -> float:
    return v if bool(rng.integers(0, 2)) else -v


def _enhance_level(level: float) -> float:
    # copied from repo: ((level/_MAX_LEVEL) * 1.8 + 0.1)
    return (level / 10.0) * 1.8 + 0.1


def _rotate_level(level: float, rng: np.random.Generator) -> float:
    # repo uses 30 deg max for _MAX_LEVEL=10, then random sign
    return _maybe_neg((level / 10.0) * 30.0, rng)


def _shear_level(level: float, rng: np.random.Generator) -> float:
    return _maybe_neg((level / 10.0) * 0.3, rng)


def _translate_level(level: float, translate_const: float, rng: np.random.Generator) -> float:
    return _maybe_neg((level / 10.0) * float(translate_const), rng)


def _apply_op_uint8(img_u8_hwc: np.ndarray, op: str, level: float, magnitude: float, rng: np.random.Generator) -> np.ndarray:
    replace = (128, 128, 128)

    if op == "Identity":
        return img_u8_hwc
    if op == "Equalize":
        return np.asarray(ImageOps.equalize(Image.fromarray(img_u8_hwc)), dtype=np.uint8)
    if op == "AutoContrast":
        return np.asarray(ImageOps.autocontrast(Image.fromarray(img_u8_hwc)), dtype=np.uint8)

    if op == "Rotate":
        deg = _rotate_level(level, rng)
        return np.asarray(Image.fromarray(img_u8_hwc).rotate(angle=deg, fillcolor=replace), dtype=np.uint8)
    if op == "TranslateX":
        px = _translate_level(level, translate_const=10, rng=rng)
        im = Image.fromarray(img_u8_hwc)
        im = im.transform(im.size, Image.AFFINE, (1, 0, px, 0, 1, 0), fillcolor=replace)
        return np.asarray(im, dtype=np.uint8)
    if op == "TranslateY":
        px = _translate_level(level, translate_const=10, rng=rng)
        im = Image.fromarray(img_u8_hwc)
        im = im.transform(im.size, Image.AFFINE, (1, 0, 0, 0, 1, px), fillcolor=replace)
        return np.asarray(im, dtype=np.uint8)
    if op == "ShearX":
        sh = _shear_level(level, rng)
        im = Image.fromarray(img_u8_hwc)
        im = im.transform(im.size, Image.AFFINE, (1, sh, 0, 0, 1, 0), Image.BICUBIC, fillcolor=replace)
        return np.asarray(im, dtype=np.uint8)
    if op == "ShearY":
        sh = _shear_level(level, rng)
        im = Image.fromarray(img_u8_hwc)
        im = im.transform(im.size, Image.AFFINE, (1, 0, 0, sh, 1, 0), Image.BICUBIC, fillcolor=replace)
        return np.asarray(im, dtype=np.uint8)

    if op in ("Brightness", "Contrast", "Color", "Sharpness"):
        factor = _enhance_level(level)
        pil = Image.fromarray(img_u8_hwc)
        if op == "Brightness":
            pil = ImageEnhance.Brightness(pil).enhance(factor)
        elif op == "Contrast":
            pil = ImageEnhance.Contrast(pil).enhance(factor)
        elif op == "Color":
            pil = ImageEnhance.Color(pil).enhance(factor)
        else:  # Sharpness
            pil = ImageEnhance.Sharpness(pil).enhance(factor)
        return np.asarray(pil, dtype=np.uint8)

    # For stain-aware ops, the repo uses args based on `magnitude` (not `level`):
    # args = magnitude * 0.03, then each op randomizes within [-factor, factor].
    if op == "Hsv":
        return hsv_shift_uint8(img_u8_hwc, factor=float(magnitude) * 0.03, rng=rng)
    if op == "Hed":
        return hed_shift_uint8(img_u8_hwc, factor=float(magnitude) * 0.03, rng=rng)

    raise ValueError(f"Unknown op: {op}")


@dataclass(frozen=True)
class HERandAugment:
    """H&E tailored RandAugment (Faryna et al., 2021).

    - Ops match the repo's `ra_type='Default'` list (includes HED/HSV shifts).
    - Per layer, magnitude is sampled as U(0, m) (paper + repo implementation).

    Args:
      n: number of sequential ops (paper suggests n=3)
      m: magnitude upper bound (paper suggests m=5)
      p: probability of applying the whole RandAugment block
    """

    n: int = 3
    m: float = 5.0
    p: float = 1.0
    ops: Sequence[str] = _DEFAULT_OPS

    def __call__(self, x_chw: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        if float(rng.random()) > float(self.p):
            return x_chw

        img_u8_hwc, scale = _to_uint8_hwc(x_chw)
        out = img_u8_hwc

        n = int(max(0, self.n))
        if n == 0 or self.m <= 0:
            return _from_uint8_hwc(out, scale=scale)

        for _ in range(n):
            op = str(self.ops[int(rng.integers(0, len(self.ops)))])
            level = float(rng.uniform(0.0, float(self.m)))
            out = _apply_op_uint8(out, op=op, level=level, magnitude=float(self.m), rng=rng)

        return _from_uint8_hwc(out, scale=scale)


@dataclass(frozen=True)
class Compose:
    """Compose numpy CHW transforms (picklable).

    ``allows_embedding_precompute`` is used by ``run_experiment``: linear probing may
    precompute backbone embeddings only when augmentations are absent or limited to
    color jitter (deterministic per epoch is not required). H&E RandAugment and
    multi-view configs force online backbone forward passes.
    """

    transforms: Sequence[Callable[[np.ndarray], np.ndarray]]
    allows_embedding_precompute: bool = True

    def __call__(self, x_chw: np.ndarray) -> np.ndarray:
        y = x_chw
        for t in self.transforms:
            y = t(y)
        return y


def make_train_augmentations(
    *,
    use_he_randaugment: bool = True,
    rand_n: int = 3,
    rand_m: float = 5.0,
    use_color_jitter: bool = True,
    jitter_brightness: float = 0.1,
    jitter_contrast: float = 0.1,
    jitter_saturation: float = 0.1,
) -> Callable[[np.ndarray], np.ndarray]:
    """Factory returning a picklable augmentation callable for `ProcessingConfig.extra_transform`."""
    ts: List[Callable[[np.ndarray], np.ndarray]] = []
    if use_color_jitter:
        ts.append(
            BasicColorJitter(
                brightness=float(jitter_brightness),
                contrast=float(jitter_contrast),
                saturation=float(jitter_saturation),
                p=1.0,
            )
        )
    if use_he_randaugment:
        ts.append(HERandAugment(n=int(rand_n), m=float(rand_m), p=1.0))
    if not ts:
        return Compose((), allows_embedding_precompute=True)
    # Jitter-only: precompute OK. Any H&E RandAugment → online forward each epoch.
    allow_pc = not bool(use_he_randaugment)
    return Compose(tuple(ts), allows_embedding_precompute=allow_pc)

