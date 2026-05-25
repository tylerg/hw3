"""Vision Transformer — §2.

You implement: PatchEmbeddings, ViT.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbeddings(nn.Module):
    """Split an image into non-overlapping patches and project each to d_model.

    Implemented with a strided Conv2d whose kernel size and stride both equal
    `patch_size`.

    Args:
        img_size:   Input image side length (assumed square). Must be divisible
                    by patch_size.
        patch_size: Side length of each patch in pixels.
        d_model:    Output embedding dimension per patch.

    Forward:
        x: (B, 3, img_size, img_size) float tensor.
        returns: (B, num_patches, d_model) where num_patches = (img_size // patch_size) ** 2.
    """

    def __init__(self, img_size: int, patch_size: int, d_model: int) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels=3,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, img_size, img_size)
        x = self.proj(x)  # (B, d_model, H/P, W/P)
        x = x.flatten(2)  # (B, d_model, N)
        x = x.transpose(1, 2)  # (B, N, d_model)
        return x


class ViT(nn.Module):
    """Vision Transformer with selectable positional encoding (learned or RoPE).

    Args:
        img_size, patch_size, d_model, num_heads, num_blocks, dropout
        pos_encoding: 'learned' (default) or 'rope'
        max_seq_len: for RoPE, maximum sequence length (patches+1)
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        d_model: int,
        num_heads: int,
        num_blocks: int,
        dropout: float = 0.1,
        pos_encoding: str = "learned",
        max_seq_len: int = 196,
        rope_base: float = 10_000.0,
    ) -> None:
        super().__init__()
        from basics.model import Block
        self.d_model = d_model
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = PatchEmbeddings(img_size, patch_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoding = pos_encoding
        self.blocks = nn.ModuleList([
            Block(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                block_size=self.num_patches + 1,
                is_decoder=False
            ) for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)

        if pos_encoding == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
        elif pos_encoding == "rope":
            from basics.rope import RoPE1D
            self.rope = RoPE1D(
                head_dim=d_model // num_heads,
                max_seq_len=max_seq_len,
                base=rope_base,
            )
        else:
            raise ValueError(f"Unknown pos_encoding: {pos_encoding}")

    def forward(self, x: torch.Tensor, return_all_tokens: bool = False) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, d_model)
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_token, x], dim=1)  # (B, N+1, d_model)

        if self.pos_encoding == "learned":
            x = x + self.pos_embed  # (B, N+1, d_model)
        elif self.pos_encoding == "rope":
            # RoPE is applied inside attention, so pass a flag to Block
            pass  # See Block implementation; RoPE is applied in attention

        for block in self.blocks:
            x = block(x, rope=self.rope if self.pos_encoding == "rope" else None)
        x = self.norm(x)
        if return_all_tokens:
            return x
        return x[:, 0]  # (B, d_model)