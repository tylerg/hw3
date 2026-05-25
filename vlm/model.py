"""Vision-Language Model — §5.

You implement: VisionLanguageModel.

Three injection strategies to support:
  - "cls":          Single visual token (the ViT's CLS embedding) prepended.
  - "all_patches":  All N+1 visual tokens (CLS + patches) prepended.
  - "interleaved":  A special <image> token in the prompt is replaced by the
                    sequence of patch embeddings at runtime.

Two attention masking strategies to support (Problem `masking`):
  - "causal":         Fully causal across the whole sequence.
  - "image_bidir":    Bidirectional within the image block, causal everywhere
                      else. Use vlm.masking.build_image_bidir_mask().
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

InjectionMode = Literal["cls", "all_patches", "interleaved"]
MaskMode = Literal["causal", "image_bidir"]


class VisionLanguageModel(nn.Module):
    """ViT image encoder + projector + pretrained causal LM decoder.

    Args:
        vit:       Your CLIP-pretrained ViT from §3.
        projector: vlm.projector.VisionLanguageProjector instance.
        decoder:   HuggingFace causal LM (e.g., SmolLM2-360M-Instruct) loaded
                   in bf16 with FlashAttention-2.
        tokenizer: Matching HF tokenizer.
        image_token_id: Token ID corresponding to the special <image> placeholder
                        in interleaved mode (None for cls / all_patches modes).

    Forward:
        images:         (B, 3, H, W) float tensor.
        input_ids:      (B, T) tokenized text.
        attention_mask: (B, T) text attention mask from the tokenizer.
        labels:         (B, T) for loss computation, or None for inference.
                        Visual-token positions must be set to -100 in labels
                        before being passed in (so they're masked out by HF's
                        loss).
        injection:      One of "cls", "all_patches", "interleaved".
        mask_mode:      One of "causal", "image_bidir".

    Returns:
        A dict with at least:
          - "loss":   scalar (only if labels was provided).
          - "logits": (B, T_total, vocab_size).
    """

    def __init__(
        self,
        vit: nn.Module,
        projector: nn.Module,
        decoder: nn.Module,
        tokenizer,
        image_token_id: int | None = None,
    ) -> None:
        super().__init__()
        self.vit = vit
        self.projector = projector
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        injection: InjectionMode = "cls",
        mask_mode: MaskMode = "causal",
    ) -> dict:
        if injection not in ("cls", "all_patches", "interleaved"):
            raise ValueError(f"Unsupported injection mode: {injection}")
        if mask_mode not in ("causal", "image_bidir"):
            raise ValueError(f"Unsupported mask mode: {mask_mode}")
        if injection == "interleaved" and self.image_token_id is None:
            raise ValueError("image_token_id must be set for interleaved injection")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        if labels is not None:
            labels = labels.clone()

        if injection == "cls":
            visual_features = self.vit(images)
        else:
            visual_features = self.vit(images, return_all_tokens=True)

        visual_embeds = self.projector(visual_features)

        dtype = next(self.decoder.parameters()).dtype

        visual_embeds = visual_embeds.to(dtype)

        text_embeds = self.decoder.get_input_embeddings()(input_ids)
        text_embeds = text_embeds.to(dtype)

        if injection == "interleaved":
            inputs_embeds, attention_mask, labels, visual_ranges = self._build_interleaved_inputs(
                text_embeds=text_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                visual_embeds=visual_embeds,
            )
            if mask_mode == "image_bidir":
                attention_mask = self._build_interleaved_bidir_mask(
                    attention_mask=attention_mask,
                    visual_ranges=visual_ranges,
                )
        else:
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            prefix_mask = torch.ones(
                visual_embeds.shape[:2],
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            if labels is not None:
                prefix_labels = torch.full(
                    (labels.size(0), visual_embeds.size(1)),
                    -100,
                    device=labels.device,
                    dtype=labels.dtype,
                )
                labels = torch.cat([prefix_labels, labels], dim=1)
            if mask_mode == "image_bidir":
                attention_mask = self._build_prefix_bidir_mask(
                    n_visual=visual_embeds.size(1),
                    n_text=text_embeds.size(1),
                    text_attention_mask=attention_mask[:, visual_embeds.size(1):],
                    device=attention_mask.device,
                    dtype=text_embeds.dtype,
                )

        outputs = self.decoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return {"loss": outputs.loss, "logits": outputs.logits}

    def _build_prefix_bidir_mask(
        self,
        n_visual: int,
        n_text: int,
        text_attention_mask: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        from vlm.masking import build_image_bidir_mask

        mask = build_image_bidir_mask(
            n_visual=n_visual,
            n_text=n_text,
            device=device,
            dtype=dtype,
        )
        if text_attention_mask is None:
            return mask

        pad_mask = text_attention_mask == 0
        if not pad_mask.any():
            return mask

        batch_size = pad_mask.shape[0]
        mask = mask.expand(batch_size, -1, -1, -1).clone()
        neg_inf = torch.finfo(dtype).min
        mask[:, :, :, n_visual:] = mask[:, :, :, n_visual:] + pad_mask.unsqueeze(1).unsqueeze(2) * neg_inf
        mask[:, :, n_visual:, :] = mask[:, :, n_visual:, :] + pad_mask.unsqueeze(1).unsqueeze(-1) * neg_inf
        return mask

    def _build_interleaved_inputs(
        self,
        text_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
        visual_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, list[list[tuple[int, int]]]]:
        batch_size, seq_len, embed_dim = text_embeds.shape
        device = text_embeds.device
        visual_length = visual_embeds.size(1)

        embedded_sequences: list[torch.Tensor] = []
        attention_sequences: list[torch.Tensor] = []
        label_sequences: list[torch.Tensor] = []
        visual_ranges: list[list[tuple[int, int]]] = []

        for batch_index in range(batch_size):
            token_ids = input_ids[batch_index]
            token_embs = text_embeds[batch_index]
            token_mask = attention_mask[batch_index]
            token_labels = labels[batch_index] if labels is not None else None

            image_positions = (token_ids == self.image_token_id).nonzero(as_tuple=True)[0]
            if image_positions.numel() == 0:
                raise ValueError(
                    "interleaved injection requires at least one image placeholder token"
                )

            pieces: list[torch.Tensor] = []
            masks: list[torch.Tensor] = []
            label_pieces: list[torch.Tensor] = []
            ranges: list[tuple[int, int]] = []
            previous_index = 0
            output_position = 0
            for image_position in image_positions.tolist():
                prefix_len = image_position - previous_index
                pieces.append(token_embs[previous_index:image_position])
                masks.append(token_mask[previous_index:image_position])
                if token_labels is not None:
                    label_pieces.append(token_labels[previous_index:image_position])

                output_position += prefix_len
                ranges.append((output_position, output_position + visual_length))
                pieces.append(visual_embeds[batch_index])
                masks.append(torch.ones(visual_length, device=device, dtype=attention_mask.dtype))
                if labels is not None:
                    label_pieces.append(
                        torch.full(
                            (visual_length,),
                            -100,
                            device=device,
                            dtype=labels.dtype,
                        )
                    )
                output_position += visual_length
                previous_index = image_position + 1

            pieces.append(token_embs[previous_index:])
            masks.append(token_mask[previous_index:])
            if token_labels is not None:
                label_pieces.append(token_labels[previous_index:])

            embedded_sequences.append(torch.cat(pieces, dim=0))
            attention_sequences.append(torch.cat(masks, dim=0))
            visual_ranges.append(ranges)
            if labels is not None:
                label_sequences.append(torch.cat(label_pieces, dim=0))

        max_len = max(sequence.size(0) for sequence in embedded_sequences)
        padded_embeds = []
        padded_masks = []
        padded_labels = []
        for batch_index in range(batch_size):
            embeddings = embedded_sequences[batch_index]
            mask = attention_sequences[batch_index]
            pad_len = max_len - embeddings.size(0)
            if pad_len > 0:
                embeddings = torch.cat(
                    [embeddings, torch.zeros(pad_len, embed_dim, device=device, dtype=embeddings.dtype)],
                    dim=0,
                )
                mask = torch.cat(
                    [mask, torch.zeros(pad_len, device=device, dtype=mask.dtype)],
                    dim=0,
                )
                if labels is not None:
                    label_sequences[batch_index] = torch.cat(
                        [
                            label_sequences[batch_index],
                            torch.full(
                                (pad_len,),
                                -100,
                                device=device,
                                dtype=labels.dtype,
                            ),
                        ],
                        dim=0,
                    )
            padded_embeds.append(embeddings)
            padded_masks.append(mask)
            if labels is not None:
                padded_labels.append(label_sequences[batch_index])

        inputs_embeds = torch.stack(padded_embeds, dim=0)
        attention_mask = torch.stack(padded_masks, dim=0)
        labels = torch.stack(padded_labels, dim=0) if labels is not None else None
        return inputs_embeds, attention_mask, labels, visual_ranges

    def _build_interleaved_bidir_mask(
        self,
        attention_mask: torch.Tensor,
        visual_ranges: list[list[tuple[int, int]]],
    ) -> torch.Tensor:
        batch_size, seq_len = attention_mask.shape
        dtype = attention_mask.dtype if attention_mask.is_floating_point() else torch.float32
        device = attention_mask.device
        neg_inf = torch.finfo(dtype).min
        base = torch.full((seq_len, seq_len), neg_inf, device=device, dtype=dtype)
        base = torch.triu(base, diagonal=1)
        mask = base.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).clone()

        for batch_index, ranges in enumerate(visual_ranges):
            for start, end in ranges:
                mask[batch_index, :, start:end, start:end] = 0

        pad_mask = attention_mask == 0
        if pad_mask.any():
            mask = mask + pad_mask.unsqueeze(1).unsqueeze(2) * neg_inf
            mask = mask + pad_mask.unsqueeze(1).unsqueeze(-1) * neg_inf
        return mask

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompts: list[str],
        injection: InjectionMode = "cls",
        max_new_tokens: int = 32,
        **gen_kwargs,
    ) -> list[str]:
        """Generate text continuations conditioned on images + prompts.

        Useful for §5's qualitative evaluation problem (vlm_qualitative).
        """
        # TODO: implement.
        raise NotImplementedError
