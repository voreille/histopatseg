from functools import partial

import torch
import torch.nn as nn


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std = torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) / std)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class QkvWithLoRA(torch.nn.Module):
    def __init__(self, qkv, rank, alpha):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.lora_q = LoRALayer(self.dim, self.dim, rank, alpha)
        self.lora_v = LoRALayer(self.dim, self.dim, rank, alpha)

    def forward(self, x):
        qkv = self.qkv(x)
        qkv[:, :, : self.dim] += self.lora_q(x)
        qkv[:, :, -self.dim :] += self.lora_v(x)
        return qkv


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def inject_lora(model: nn.Module, rank: int = 8, alpha: float = 1.0) -> nn.Module:
    """
    Injects LoRA adapters into a timm ViT model's qkv projection layers.

    Args:
        model (nn.Module): The ViT model (from timm) to modify.
        rank (int): LoRA rank.
        alpha (float): LoRA alpha (scaling factor).

    Returns:
        nn.Module: Modified model with LoRA adapters injected.
    """

    # Make sure it's a ViT model with .blocks and .attn.qkv
    assert hasattr(model, "blocks"), "Model does not have transformer blocks."

    # Inject LoRA into qkv projections
    assign_lora = partial(QkvWithLoRA, rank=rank, alpha=alpha)
    for block in model.blocks:
        block.attn.qkv = assign_lora(block.attn.qkv)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only lora_q, lora_v, and classifier head
    for block in model.blocks:
        for param in block.attn.qkv.lora_q.parameters():
            param.requires_grad = True
        for param in block.attn.qkv.lora_v.parameters():
            param.requires_grad = True

    if hasattr(model, "head"):  # timm ViT classifier head
        for param in model.head.parameters():
            param.requires_grad = True

    return model
