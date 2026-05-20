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


class Affine(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.alpha + self.beta


class ResMLPBlock(nn.Module):
    def __init__(self, num_patches: int, embed_dim: int, expansion: int = 4) -> None:
        super().__init__()
        self.affine1 = Affine(embed_dim)
        self.token_mixer = nn.Linear(num_patches, num_patches)
        self.affine2 = Affine(embed_dim)
        hidden_dim = embed_dim * expansion
        self.channel_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.affine1(x).transpose(1, 2)
        y = self.token_mixer(y).transpose(1, 2)
        x = x + y
        x = x + self.channel_mlp(self.affine2(x))
        return x


class TinyResMLP(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        num_classes: int = 10,
        embed_dim: int = 256,
        depth: int = 8,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = PatchEmbed(3, embed_dim, patch_size)
        self.blocks = nn.Sequential(
            *[
                ResMLPBlock(
                    num_patches=num_patches,
                    embed_dim=embed_dim,
                    expansion=expansion,
                )
                for _ in range(depth)
            ]
        )
        self.affine = Affine(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.affine(x).mean(dim=1)
        return self.head(x)
