"""Tests for §5 — Vision-Language Model."""

from __future__ import annotations

import torch
import torch.nn as nn

from basics.vit import ViT
from vlm.model import VisionLanguageModel
from vlm.projector import VisionLanguageProjector


class DummyVit(nn.Module):
    def __init__(self, d_image: int, n_tokens: int) -> None:
        super().__init__()
        self.d_model = d_image
        self.n_tokens = n_tokens

    def forward(self, x: torch.Tensor, return_all_tokens: bool = False) -> torch.Tensor:
        B = x.shape[0]
        if return_all_tokens:
            return torch.randn(B, self.n_tokens, self.d_model, device=x.device, dtype=x.dtype)
        return torch.randn(B, self.d_model, device=x.device, dtype=x.dtype)


class DummyDecoder(nn.Module):
    def __init__(self, d_decoder: int, vocab_size: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_decoder)
        self.head = nn.Linear(d_decoder, vocab_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor | None = None, labels: torch.Tensor | None = None):
        logits = self.head(inputs_embeds)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return type("Output", (), {"loss": loss, "logits": logits})


def test_vit_return_all_tokens():
    model = ViT(img_size=32, patch_size=8, d_model=16, num_heads=4, num_blocks=1, dropout=0.0)
    x = torch.randn(2, 3, 32, 32)
    out = model(x, return_all_tokens=True)
    expected_tokens = (32 // 8) ** 2 + 1
    assert out.shape == (2, expected_tokens, 16)


def test_vlm_forward_injection_modes():
    vit = DummyVit(d_image=16, n_tokens=3)
    projector = VisionLanguageProjector(d_image=16, d_decoder=32, expansion=2)
    decoder = DummyDecoder(d_decoder=32, vocab_size=16)
    model = VisionLanguageModel(vit, projector, decoder, tokenizer=None, image_token_id=0)

    images = torch.randn(2, 3, 64, 64)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    out_cls = model(
        images,
        input_ids,
        attention_mask,
        labels=labels,
        injection="cls",
        mask_mode="causal",
    )
    assert out_cls["logits"].shape == (2, 4, 16)
    assert out_cls["loss"] is not None

    out_all = model(
        images,
        input_ids,
        attention_mask,
        labels=labels,
        injection="all_patches",
        mask_mode="causal",
    )
    assert out_all["logits"].shape == (2, 6, 16)
    assert out_all["loss"] is not None

    interleaved_ids = torch.tensor([[0, 1, 2], [1, 0, 2]], dtype=torch.long)
    interleaved_attention = torch.ones_like(interleaved_ids)
    out_interleaved = model(
        images,
        interleaved_ids,
        interleaved_attention,
        labels=labels,
        injection="interleaved",
        mask_mode="causal",
    )
    assert out_interleaved["logits"].shape == (2, 5, 16)
    assert out_interleaved["loss"] is not None
