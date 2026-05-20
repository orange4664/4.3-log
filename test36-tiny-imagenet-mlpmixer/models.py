from __future__ import annotations

import torch
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int = 3, embed_dim: int = 320, patch_size: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class MlpBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, num_patches: int, embed_dim: int, token_dim: int, channel_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_tokens = nn.LayerNorm(embed_dim)
        self.token_mlp = MlpBlock(num_patches, token_dim, dropout)
        self.norm_channels = nn.LayerNorm(embed_dim)
        self.channel_mlp = MlpBlock(embed_dim, channel_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm_tokens(x).transpose(1, 2)
        y = self.token_mlp(y).transpose(1, 2)
        x = x + y
        x = x + self.channel_mlp(self.norm_channels(x))
        return x


class TinyMLPMixer(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 4,
        num_classes: int = 200,
        embed_dim: int = 320,
        depth: int = 8,
        token_dim: int = 160,
        channel_dim: int = 640,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = PatchEmbed(3, embed_dim, patch_size)
        self.blocks = nn.Sequential(
            *[
                MixerBlock(
                    num_patches=num_patches,
                    embed_dim=embed_dim,
                    token_dim=token_dim,
                    channel_dim=channel_dim,
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
