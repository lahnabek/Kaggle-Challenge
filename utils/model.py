"""Frozen backbones (DINOv2 / HF timm), default adapter/head, and full model assembly."""

from __future__ import annotations

from typing import List, Optional, Literal

import torch
import torch.nn as nn

from utils.config import ModelConfig
from utils.constants import DEVICE


class Virchow2Wrapper(nn.Module):
    """Wrap Virchow2 timm model to return a single embedding per image."""

    def __init__(self, base: nn.Module, pool: Literal["cls", "cls_mean"]) -> None:
        super().__init__()
        self.base = base
        self.pool = pool
        if pool == "cls":
            self.num_features = 1280
        elif pool == "cls_mean":
            self.num_features = 2560
        else:
            raise ValueError(f"Unknown Virchow2 pooling mode: {pool!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)  # expected (B, T, 1280)
        cls = out[:, 0]
        if self.pool == "cls":
            return cls
        patch = out[:, 5:]  # skip register tokens 1-4
        return torch.cat([cls, patch.mean(1)], dim=-1)


def _freeze(m: nn.Module) -> nn.Module:
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def load_frozen_backbone(backbone_name: str, device: Optional[torch.device] = None) -> nn.Module:
    """Load a frozen backbone.

    Supported values:
    - DINOv2 via torch.hub: e.g. "dinov2_vits14"
    - timm/HF: "uni", "uni2_h", "virchow2_cls", "virchow2_clsmean"
    """
    dev = device or DEVICE

    if backbone_name.startswith("dinov2_"):
        backbone = torch.hub.load("facebookresearch/dinov2", backbone_name).to(dev)
        return _freeze(backbone)

    if backbone_name == "uni":
        import timm

        backbone = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            num_classes=0,
            init_values=1e-5,
            dynamic_img_size=True,
        ).to(dev)
        return _freeze(backbone)

    if backbone_name == "uni2_h":
        import timm
        import torch as _torch

        # Parameters adapted from the UNI2-h reference recipe (kept explicit on purpose).
        backbone = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h",
            pretrained=True,
            num_classes=0,
            img_size=224,
            patch_size=14,
            depth=24,
            num_heads=24,
            embed_dim=1536,
            mlp_ratio=2.66667 * 2,
            init_values=1e-5,
            no_embed_class=True,
            reg_tokens=8,
            dynamic_img_size=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=_torch.nn.SiLU,
        ).to(dev)
        return _freeze(backbone)

    if backbone_name in ("virchow2_cls", "virchow2_clsmean"):
        import timm
        import torch as _torch
        from timm.layers import SwiGLUPacked

        base = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=_torch.nn.SiLU,
        ).to(dev)
        pool: Literal["cls", "cls_mean"] = "cls" if backbone_name == "virchow2_cls" else "cls_mean"
        backbone = Virchow2Wrapper(base=base, pool=pool).to(dev)
        return _freeze(backbone)

    raise ValueError(f"Unknown backbone_name: {backbone_name!r}")


class DefaultMLPAdapter(nn.Module):
    """Default adapter: bottleneck MLP residual on the embedding."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.down = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(hidden_dim, in_dim)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.up(self.drop(self.act(self.down(x))))


class DefaultBinaryHead(nn.Module):
    """Default head: linear logits of shape (B, 1)."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class FullModel(nn.Module):
    """Backbone + optional adapter + classification head."""

    def __init__(self, backbone: nn.Module, model_cfg: ModelConfig) -> None:
        super().__init__()
        self.backbone = backbone

        if not hasattr(backbone, "num_features"):
            raise ValueError("Backbone is expected to expose `num_features`.")
        in_dim = int(backbone.num_features)

        self.adapter = None
        if model_cfg.adapter.enabled:
            cls = model_cfg.adapter.module_cls or DefaultMLPAdapter
            kwargs = model_cfg.adapter.module_kwargs or {}
            self.adapter = cls(in_dim=in_dim, **kwargs)

        if not model_cfg.head.enabled:
            raise ValueError("Head must be enabled for classification.")
        head_cls = model_cfg.head.module_cls or DefaultBinaryHead
        head_kwargs = model_cfg.head.module_kwargs or {}
        self.head = head_cls(in_dim=in_dim, **head_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        if self.adapter is not None:
            z = self.adapter(z)
        return self.head(z)


def trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Parameters with ``requires_grad`` (typically adapter + head)."""
    return [p for p in model.parameters() if p.requires_grad]


def build_binary_head(model_cfg: ModelConfig, in_dim: int) -> nn.Module:
    """Instantiate the classification head (same rules as ``FullModel``)."""
    if not model_cfg.head.enabled:
        raise ValueError("Head must be enabled.")
    head_cls = model_cfg.head.module_cls or DefaultBinaryHead
    kwargs = model_cfg.head.module_kwargs or {}
    return head_cls(in_dim=in_dim, **kwargs)


class HeadOnly(nn.Module):
    """Linear probing on precomputed backbone embeddings (no adapter)."""

    def __init__(self, head: nn.Module) -> None:
        super().__init__()
        self.head = head

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)
