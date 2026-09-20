"""Shared output heads and model wrapper for the ``archs/`` modules.

Every arch model exposes:
    ARCH: str                       — registry name
    CFG: dict                       — default config (saved into ckpt["model_cfg"])
    build(cfg: dict | None) -> nn.Module
and the module maps (B, 3, L) -> {"map_native": (B, L), "conf_logit": (B,)}
for ARBITRARY integer L (no fixed-length component anywhere).

All ops are PyTorch-official ATen (Conv1d/GroupNorm/LayerNorm/GELU/Linear/fft)
so torch.compile can ingest the whole forward (fullgraph target).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MapConfHeads(nn.Module):
    """Native-L map head + pooled confidence head."""

    def __init__(self, ch: int, gn_groups: int):
        super().__init__()
        self.head_map = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(gn_groups, ch),
            nn.GELU(),
            nn.Conv1d(ch, 1, 1),
        )
        self.head_conf = nn.Sequential(
            nn.Linear(2 * ch, ch), nn.GELU(), nn.Linear(ch, 1),
        )

    def forward(self, h):
        map_native = self.head_map(h).squeeze(1)
        conf = self.head_conf(
            torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)).squeeze(-1)
        return {"map_native": map_native, "conf_logit": conf}


class ArchModel(nn.Module):
    """trunk(B,3,L)->(B,C,L) plus shared heads."""

    def __init__(self, trunk: nn.Module, ch: int, gn_groups: int):
        super().__init__()
        self.trunk = trunk
        self.heads = MapConfHeads(ch, gn_groups)

    def forward(self, x):
        return self.heads(self.trunk(x))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
