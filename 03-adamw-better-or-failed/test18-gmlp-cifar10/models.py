from __future__ import annotations

import torch
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int = 3, embed_dim: int = 256, patch_size: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class SpatialGatingUnit(nn.Module):
    def __init__(self, num_patches: int, hidden_dim: int) -> None:
        super().__init__()
        gated_dim = hidden_dim // 2
        self.norm = nn.LayerNorm(gated_dim)
        self.proj = nn.Linear(num_patches, num_patches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v).transpose(1, 2)
        v = self.proj(v).transpose(1, 2)
        return u * v


class GmlpBlock(nn.Module):
    def __init__(self, num_patches: int, embed_dim: int, ff_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.act = nn.GELU()
        self.sgu = SpatialGatingUnit(num_patches, ff_dim)
        self.fc2 = nn.Linear(ff_dim // 2, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.sgu(y)
        y = self.fc2(y)
        y = self.drop(y)
        return x + y


class TinyGMLP(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        num_classes: int = 10,
        embed_dim: int = 256,
        depth: int = 6,
        ff_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = PatchEmbed(3, embed_dim, patch_size)
        self.blocks = nn.Sequential(
            *[
                GmlpBlock(
                    num_patches=num_patches,
                    embed_dim=embed_dim,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x.mean(dim=1))
        return self.head(x)
