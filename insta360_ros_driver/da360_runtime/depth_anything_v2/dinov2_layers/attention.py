# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


logger = logging.getLogger("dinov2")


try:
    from xformers.ops import memory_efficient_attention, unbind, fmha
    from xformers.ops.fmha import cutlass

    # The default xFormers operator selects FlashAttention on newer GPUs, but
    # the prebuilt wheel may not contain a Flash kernel for Blackwell (sm_120).
    # CUTLASS is available in the wheel and supports this device.
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 12:
        XFORMERS_ATTENTION_OP = (cutlass.FwOp, cutlass.BwOp)
    else:
        XFORMERS_ATTENTION_OP = None

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False
    XFORMERS_ATTENTION_OP = None


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]
        # The legacy ONNX exporter cannot serialize the ``scale`` keyword on
        # scaled_dot_product_attention reliably.  During tracing use the
        # equivalent explicit matmul/softmax path; normal CUDA inference keeps
        # the fused PyTorch kernel.
        if hasattr(F, 'scaled_dot_product_attention') and not torch.jit.is_tracing():
            dropout_p = self.attn_drop.p if self.training else 0.0
            try:
                x = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=dropout_p,
                    is_causal=False,
                    scale=self.scale,
                )
            except TypeError:
                # PyTorch versions before the scale keyword are still usable.
                x = F.scaled_dot_product_attention(
                    q * self.scale,
                    k,
                    v,
                    dropout_p=dropout_p,
                    is_causal=False,
                )
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        attention_kwargs = {"attn_bias": attn_bias}
        if XFORMERS_ATTENTION_OP is not None:
            attention_kwargs["op"] = XFORMERS_ATTENTION_OP
        x = memory_efficient_attention(q, k, v, **attention_kwargs)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

        
