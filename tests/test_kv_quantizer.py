"""Tests for KVQuantizer INT8/INT4 quantization."""

import pytest
import torch

from vllm_kv_shrinker.core.kv_quantizer import (
    KVQuantizer,
    QuantizedKV,
    combined_compression_ratio,
)
from vllm_kv_shrinker.core.config import KVShrinkerConfig, EvictionPolicy
from vllm_kv_shrinker.core.kv_shrinker import KVShrinker


def make_kv(tokens=32, heads=4, head_dim=64, seed=0):
    torch.manual_seed(seed)
    return torch.randn(tokens, heads, head_dim)


# ---------------------------------------------------------------------------
# INT8
# ---------------------------------------------------------------------------

class TestInt8Quantizer:
    def test_output_dtype(self):
        q = KVQuantizer(bits=8)
        kv = make_kv()
        out = q.quantize(kv)
        assert out.data.dtype == torch.int8
        assert out.bits == 8

    def test_output_shape(self):
        q = KVQuantizer(bits=8)
        kv = make_kv(tokens=16, heads=8, head_dim=64)
        out = q.quantize(kv)
        assert out.data.shape == (16, 8, 64)
        assert out.scale.shape == (16, 8, 1)

    def test_dequantize_close(self):
        q = KVQuantizer(bits=8)
        kv = make_kv()
        out = q.quantize(kv)
        recon = out.dequantize()
        assert recon.shape == kv.shape
        # INT8 absmax: max relative error should be small
        assert (recon - kv).abs().max().item() < 0.1

    def test_quantize_and_dequantize_dtype_preserved(self):
        q = KVQuantizer(bits=8)
        kv = make_kv().half()
        recon = q.quantize_and_dequantize(kv)
        assert recon.dtype == torch.float16

    def test_compression_ratio(self):
        q = KVQuantizer(bits=8)
        kv = make_kv()
        out = q.quantize(kv)
        # INT8 data = half the bytes of FP16, plus small scale overhead
        assert out.compression_ratio > 1.5

    def test_invalid_bits(self):
        with pytest.raises(ValueError):
            KVQuantizer(bits=3)


# ---------------------------------------------------------------------------
# INT4
# ---------------------------------------------------------------------------

class TestInt4Quantizer:
    def test_output_dtype(self):
        q = KVQuantizer(bits=4, group_size=64)
        kv = make_kv(head_dim=64)
        out = q.quantize(kv)
        assert out.data.dtype == torch.uint8
        assert out.bits == 4

    def test_packed_shape(self):
        # head_dim=64 packed → 32 bytes per head per token
        q = KVQuantizer(bits=4, group_size=64)
        kv = make_kv(tokens=16, heads=4, head_dim=64)
        out = q.quantize(kv)
        assert out.data.shape == (16, 4, 32)  # head_dim // 2

    def test_scale_shape(self):
        q = KVQuantizer(bits=4, group_size=64)
        kv = make_kv(tokens=16, heads=4, head_dim=64)
        out = q.quantize(kv)
        # 1 group per head (64 // 64 = 1)
        assert out.scale.shape == (16, 4, 1)

    def test_dequantize_close(self):
        q = KVQuantizer(bits=4, group_size=64)
        kv = make_kv()
        out = q.quantize(kv)
        recon = out.dequantize()
        assert recon.shape == kv.shape
        # INT4 is coarser — allow larger error
        assert (recon - kv).abs().mean().item() < 0.2

    def test_roundtrip_shape_preserved(self):
        q = KVQuantizer(bits=4, group_size=128)
        kv = make_kv(tokens=8, heads=2, head_dim=128)
        out = q.quantize(kv)
        recon = out.dequantize()
        assert recon.shape == kv.shape

    def test_group_size_mismatch_raises(self):
        q = KVQuantizer(bits=4, group_size=100)  # 64 not divisible by 100
        kv = make_kv(head_dim=64)
        with pytest.raises(AssertionError):
            q.quantize(kv)

    def test_compression_ratio_greater_than_int8(self):
        q8 = KVQuantizer(bits=8)
        q4 = KVQuantizer(bits=4, group_size=64)
        kv = make_kv(head_dim=64)
        r8 = q8.quantize(kv).compression_ratio
        r4 = q4.quantize(kv).compression_ratio
        assert r4 > r8


# ---------------------------------------------------------------------------
# KVShrinker integration
# ---------------------------------------------------------------------------

class TestKVShrinkerWithQuant:
    def _make_inputs(self, seq_len=64):
        torch.manual_seed(42)
        key = torch.randn(seq_len, 4, 64)
        val = torch.randn(seq_len, 4, 64)
        raw = torch.rand(8, seq_len, seq_len)
        attn = raw / raw.sum(dim=-1, keepdim=True)
        return key, val, attn

    def test_compress_with_int8(self):
        config = KVShrinkerConfig(budget_ratio=0.5, policy=EvictionPolicy.SNAPKV, quant_bits=8)
        shrinker = KVShrinker(config)
        key, val, attn = self._make_inputs()
        pk, pv, mask = shrinker.compress(key, val, attn)
        assert pk.dtype == key.dtype
        assert pk.shape[0] == mask.sum().item()

    def test_compress_with_int4(self):
        config = KVShrinkerConfig(
            budget_ratio=0.5, policy=EvictionPolicy.SNAPKV,
            quant_bits=4, quant_group_size=64,
        )
        shrinker = KVShrinker(config)
        key, val, attn = self._make_inputs()
        pk, pv, mask = shrinker.compress(key, val, attn)
        assert pk.shape[0] == mask.sum().item()

    def test_compress_quantized_returns_quantizedkv(self):
        config = KVShrinkerConfig(budget_ratio=0.5, quant_bits=8)
        shrinker = KVShrinker(config)
        key, val, attn = self._make_inputs()
        qk, qv, mask = shrinker.compress_quantized(key, val, attn)
        assert isinstance(qk, QuantizedKV)
        assert isinstance(qv, QuantizedKV)
        assert qk.bits == 8

    def test_compress_quantized_requires_quant_bits(self):
        config = KVShrinkerConfig(budget_ratio=0.5)  # no quant_bits
        shrinker = KVShrinker(config)
        key, val, attn = self._make_inputs()
        with pytest.raises(RuntimeError):
            shrinker.compress_quantized(key, val, attn)

    def test_dequantized_output_similar_to_unquantized(self):
        """Quantization error should be small for INT8."""
        config_q = KVShrinkerConfig(budget_ratio=0.5, policy=EvictionPolicy.SNAPKV, quant_bits=8)
        config_nq = KVShrinkerConfig(budget_ratio=0.5, policy=EvictionPolicy.SNAPKV)
        sq = KVShrinker(config_q)
        snq = KVShrinker(config_nq)
        key, val, attn = self._make_inputs()
        pk_q, _, mask_q = sq.compress(key, val, attn)
        pk_nq, _, mask_nq = snq.compress(key, val, attn)
        # Same tokens kept (same policy, same seed)
        assert (mask_q == mask_nq).all()
        assert (pk_q - pk_nq).abs().max().item() < 0.1


# ---------------------------------------------------------------------------
# combined_compression_ratio utility
# ---------------------------------------------------------------------------

class TestCombinedCompressionRatio:
    def test_no_quant(self):
        # 30% budget, no quant → 1/0.3 ≈ 3.33×
        r = combined_compression_ratio(0.3, None)
        assert abs(r - (1 / 0.3)) < 0.01

    def test_int8(self):
        # 30% budget + INT8 → 2 × (1/0.3) ≈ 6.67×
        r = combined_compression_ratio(0.3, 8)
        assert abs(r - 2 / 0.3) < 0.01

    def test_int4(self):
        # 30% budget + INT4 → 4 × (1/0.3) ≈ 13.3×
        r = combined_compression_ratio(0.3, 4)
        assert abs(r - 4 / 0.3) < 0.01
