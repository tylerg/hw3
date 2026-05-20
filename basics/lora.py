"""LoRA adapters — §4.

You implement: LoRALinear, apply_lora_to_attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping an existing nn.Linear layer."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        # 1. Keep the base layer and freeze its parameters
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        self.rank = rank
        self.alpha = alpha
        
        # 2. Initialize trainable low-rank matrices A and B
        self.A = nn.Parameter(torch.empty(rank, base_layer.in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        
        self.B = nn.Parameter(torch.zeros(base_layer.out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns base_layer(x) + (alpha / rank) * (x @ A.T @ B.T)
        return self.base_layer(x) + (self.alpha / self.rank) * (x @ self.A.T @ self.B.T)


def apply_lora_to_attention(model: nn.Module, rank: int, alpha: float) -> nn.Module:
    """Replace `q_proj` and `v_proj` linear layers in every attention head
    with LoRA-wrapped versions.
    """
    
    # CRITICAL FIX: Freeze the *entire* model first before injecting adapters.
    # Otherwise, MLPs and LayerNorms remain trainable, causing the >86% error.
    for param in model.parameters():
        param.requires_grad = False

    from basics.model import Head
    for module_name, module in model.named_modules():
        if isinstance(module, Head):
            if hasattr(module, "q_proj") and isinstance(module.q_proj, nn.Linear):
                setattr(module, "q_proj", LoRALinear(module.q_proj, rank, alpha))
            if hasattr(module, "v_proj") and isinstance(module.v_proj, nn.Linear):
                setattr(module, "v_proj", LoRALinear(module.v_proj, rank, alpha))
                
    return model


def print_param_stats(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total}")
    print(f"Trainable parameters: {trainable}")
    print(f"Trainable ratio: {trainable / total:.4f}")