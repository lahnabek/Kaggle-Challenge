"""Frozen DINOv2 backbone, default adapter/head, and full model assembly."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from utils.config import ModelConfig
from utils.constants import DEVICE


def load_frozen_backbone(backbone_name: str, device: Optional[torch.device] = None) -> nn.Module:
    """Load a DINOv2 backbone from torch.hub and freeze parameters."""
    dev = device or DEVICE
    backbone = torch.hub.load("facebookresearch/dinov2", backbone_name).to(dev)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


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
