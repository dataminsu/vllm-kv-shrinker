"""
KV Cache Quantizer — post-eviction INT8/INT4 compression.

Orthogonal to token pruning: applied after KVShrinker selects which tokens
to keep.  Combined savings = budget_ratio × (quant_bits / 16).

Quantization schemes
--------------------
INT8 : signed absmax per-head-per-token.
       scale shape : [tokens, heads, 1]
       data shape  : [tokens, heads, head_dim]  dtype=int8

INT4 : signed grouped absmax (group_size=128, aligned with typical head_dim).
       scale shape : [tokens, heads, groups]     dtype=float16
       data shape  : [tokens, heads, head_dim//2] dtype=uint8  (2×4-bit packed)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class QuantizedKV:
    """Container for a quantized KV tensor with its dequantization metadata."""

    data: torch.Tensor          # int8 (INT8) or uint8 packed (INT4)
    scale: torch.Tensor         # float32/float16 scale factors
    bits: int                   # 8 or 4
    original_shape: tuple       # shape before quantization
    original_dtype: torch.dtype
    group_size: int = 128       # used only for INT4

    # ------------------------------------------------------------------
    # Memory accounting
    # ------------------------------------------------------------------

    @property
    def quantized_bytes(self) -> int:
        """Bytes consumed by the quantized representation (data + scale)."""
        data_bytes = self.data.numel() * (self.bits // 8)
        scale_bytes = self.scale.numel() * 2  # float16 scales
        return data_bytes + scale_bytes

    @property
    def original_bytes(self) -> int:
        """Bytes consumed by the original FP16 tensor."""
        n = 1
        for d in self.original_shape:
            n *= d
        return n * 2  # float16

    @property
    def compression_ratio(self) -> float:
        """original_bytes / quantized_bytes."""
        return self.original_bytes / self.quantized_bytes

    # ------------------------------------------------------------------
    # Dequantization
    # ------------------------------------------------------------------

    def dequantize(self) -> torch.Tensor:
        """Reconstruct a float32 tensor from the quantized representation."""
        if self.bits == 8:
            return _dequantize_int8(self.data, self.scale, self.original_dtype)
        elif self.bits == 4:
            return _dequantize_int4(
                self.data, self.scale, self.original_shape,
                self.group_size, self.original_dtype,
            )
        else:
            raise ValueError(f"Unsupported bits={self.bits}")


# ---------------------------------------------------------------------------
# KVQuantizer
# ---------------------------------------------------------------------------

class KVQuantizer:
    """
    Quantizes KV tensors to INT8 or INT4 after token eviction.

    Args:
        bits:       Target bit-width (8 or 4).
        group_size: Grouping for INT4 scale factors (default 128).
                    Must evenly divide head_dim.
    """

    SUPPORTED_BITS = (8, 4)

    def __init__(self, bits: int = 8, group_size: int = 128) -> None:
        if bits not in self.SUPPORTED_BITS:
            raise ValueError(f"bits must be one of {self.SUPPORTED_BITS}, got {bits}")
        self.bits = bits
        self.group_size = group_size

    def quantize(self, x: torch.Tensor) -> QuantizedKV:
        """
        Quantize a [tokens, heads, head_dim] float tensor.

        Args:
            x: Float32 or Float16 tensor of shape [tokens, heads, head_dim].

        Returns:
            QuantizedKV with dequantization metadata.
        """
        original_shape = tuple(x.shape)
        original_dtype = x.dtype
        x_f = x.float()  # promote to float32 for stable scale computation

        if self.bits == 8:
            data, scale = _quantize_int8(x_f)
        else:
            data, scale = _quantize_int4(x_f, self.group_size)

        return QuantizedKV(
            data=data,
            scale=scale.half(),  # store scales in float16
            bits=self.bits,
            original_shape=original_shape,
            original_dtype=original_dtype,
            group_size=self.group_size,
        )

    def quantize_and_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize then immediately dequantize — useful for measuring
        quantization error without changing the downstream tensor dtype.
        """
        return self.quantize(x).dequantize().to(x.dtype)

    @property
    def theoretical_compression_vs_fp16(self) -> float:
        """
        Compression of data weights alone (ignoring scale overhead).
        INT8 → 2×, INT4 → 4×.
        """
        return 16 / self.bits


# ---------------------------------------------------------------------------
# INT8 primitives
# ---------------------------------------------------------------------------

def _quantize_int8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Signed absmax INT8 quantization, per-head-per-token.

    x    : [tokens, heads, head_dim]
    scale: [tokens, heads, 1]        float32
    data : [tokens, heads, head_dim] int8
    """
    # Compute per-row scale
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / 127.0
    data = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return data, scale


def _dequantize_int8(
    data: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return (data.float() * scale.float()).to(out_dtype)


# ---------------------------------------------------------------------------
# INT4 primitives  (grouped absmax, 2×4-bit packed into uint8)
# ---------------------------------------------------------------------------

def _quantize_int4(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Grouped signed absmax INT4, packed 2 values per byte.

    x      : [tokens, heads, head_dim]
    scale  : [tokens, heads, head_dim // group_size]  float32
    data   : [tokens, heads, head_dim // 2]           uint8 (packed)

    INT4 range: -8 … 7  (signed 4-bit two's complement)
    """
    tokens, heads, head_dim = x.shape
    assert head_dim % group_size == 0, (
        f"head_dim ({head_dim}) must be divisible by group_size ({group_size})"
    )
    n_groups = head_dim // group_size

    # Reshape to groups: [tokens, heads, n_groups, group_size]
    xg = x.reshape(tokens, heads, n_groups, group_size)

    # Per-group absmax scale
    amax = xg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)  # [T, H, G, 1]
    scale = (amax / 7.0).squeeze(-1)  # [tokens, heads, n_groups]

    # Quantize to int4 range [-8, 7]
    q = (xg / amax * 7.0).round().clamp(-8, 7).to(torch.int8)
    q_flat = q.reshape(tokens, heads, head_dim)  # [T, H, D]

    # Pack two int4 values into one uint8 byte
    # Even indices → low nibble, odd indices → high nibble
    low = q_flat[..., 0::2]   # [T, H, D//2]
    high = q_flat[..., 1::2]  # [T, H, D//2]
    # Mask to 4 bits and pack
    packed = ((high.to(torch.uint8) & 0x0F) << 4) | (low.to(torch.uint8) & 0x0F)

    return packed, scale


def _dequantize_int4(
    data: torch.Tensor,
    scale: torch.Tensor,
    original_shape: tuple,
    group_size: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Unpack and dequantize INT4 data."""
    tokens, heads, head_dim = original_shape

    # Unpack nibbles
    low_u = (data & 0x0F).to(torch.int8)   # [T, H, D//2]
    high_u = ((data >> 4) & 0x0F).to(torch.int8)

    # Sign-extend 4-bit to int8: values > 7 are negative
    low = _sign_extend_4bit(low_u)
    high = _sign_extend_4bit(high_u)

    # Interleave back to [T, H, D]
    q = torch.empty(tokens, heads, head_dim, dtype=torch.float32, device=data.device)
    q[..., 0::2] = low.float()
    q[..., 1::2] = high.float()

    # Dequantize via per-group scale
    n_groups = head_dim // group_size
    q_grouped = q.reshape(tokens, heads, n_groups, group_size)
    s = scale.float().unsqueeze(-1)  # [T, H, n_groups, 1]
    out = (q_grouped * s).reshape(tokens, heads, head_dim)

    return out.to(out_dtype)


def _sign_extend_4bit(x: torch.Tensor) -> torch.Tensor:
    """Sign-extend unsigned 4-bit values stored in int8 to signed int8."""
    # Values 8-15 (0x8-0xF) are negative in 4-bit two's complement
    return torch.where(x > 7, x - 16, x.to(torch.int8))


# ---------------------------------------------------------------------------
# Convenience: compute combined compression ratio
# ---------------------------------------------------------------------------

def combined_compression_ratio(
    budget_ratio: float,
    quant_bits: Optional[int],
    fp16_baseline: bool = True,
) -> float:
    """
    Total memory reduction from token pruning + optional quantization.

    Args:
        budget_ratio: Fraction of tokens kept (e.g. 0.3).
        quant_bits:   Quantization bit-width (8 or 4), or None for FP16.
        fp16_baseline: If True, assumes FP16 original (16 bits).

    Returns:
        compression_ratio: e.g. 10.7 means 10.7× smaller than original FP16.
    """
    baseline_bits = 16 if fp16_baseline else 32
    quant = quant_bits if quant_bits is not None else baseline_bits
    return (baseline_bits / quant) / budget_ratio
