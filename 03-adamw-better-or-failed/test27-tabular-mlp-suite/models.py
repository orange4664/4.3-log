from __future__ import annotations

from typing import List

import torch
from torch import nn


class TabularMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        width: int = 512,
        depth: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        current = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(current, width))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(width))
            layers.append(nn.Dropout(dropout))
            current = width
        layers.append(nn.Linear(current, num_classes))
        self.net = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
