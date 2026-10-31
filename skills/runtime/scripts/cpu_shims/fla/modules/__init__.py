from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, elementwise_affine: bool = True, bias: bool = False,
                 eps: float = 1e-5, **kwargs):
        super().__init__()
        self.hidden_size, self.eps = hidden_size, eps
        self.weight = nn.Parameter(torch.ones(hidden_size)) if elementwise_affine else None
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if (elementwise_affine and bias) else None

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        if residual is not None:
            x = x + residual
            residual = x
        y = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)
        if self.weight is not None:
            y = y * self.weight
        if self.bias is not None:
            y = y + self.bias
        return (y, residual) if prenorm else y


class FusedRMSNormGated(RMSNorm):
    def __init__(self, hidden_size: int, elementwise_affine: bool = True, eps: float = 1e-5,
                 activation: str = "swish", **kwargs):
        super().__init__(hidden_size, elementwise_affine=elementwise_affine, eps=eps)
        self.activation = activation

    def forward(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
        y = super().forward(x)
        act = F.silu if self.activation in ("swish", "silu") else torch.sigmoid
        return y * act(g)


class ShortConvolution(nn.Conv1d):
    """Causal depthwise conv over (B, T, D); returns (y, cache)."""

    def __init__(self, hidden_size: int, kernel_size: int, bias: bool = False,
                 activation: Optional[str] = "silu", use_fast_conv1d: bool = True, **kwargs):
        super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size,
                         padding=kernel_size - 1, bias=bias)
        self.hidden_size, self.activation = hidden_size, activation

    def forward(self, x, mask=None, cache=None, output_final_state=False, cu_seqlens=None, **kwargs):
        if mask is not None:
            x = x.mul(mask.unsqueeze(-1))
        T = x.shape[1]
        y = super().forward(x.transpose(1, 2))[..., :T].transpose(1, 2)
        if self.activation in ("silu", "swish"):
            y = F.silu(y)
        new_cache = None
        if output_final_state:
            k = self.kernel_size[0]
            xt = x.transpose(1, 2)
            new_cache = F.pad(xt, (max(0, k - T), 0))[..., -k:]
        return y, new_cache


def l2norm(x, eps: float = 1e-6, **kwargs):
    return x * torch.rsqrt(x.float().pow(2).sum(-1, keepdim=True) + eps).to(x.dtype)
