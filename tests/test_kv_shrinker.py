"""Integration tests for KVShrinker end-to-end."""

import pytest
import torch

from vllm_kv_shrinker.core.kv_shrinker import KVShrinker
from vllm_kv_shrinker.core.config import KVShrinkerConfig, EvictionPolicy
from vllm_kv_shrinker.rag.rag_signal import RAGSignal


def make_tensors(seq_len=128, num_heads=8, num_kv_heads=4, head_dim=64):
    torch.manual_seed(99)
    key = torch.randn(seq_len, num_kv_heads, head_dim)
    val = torch.randn(seq_len, num_kv_heads, head_dim)
    # Attention weights: [num_heads, q_len=seq_len, k_len=seq_len]
    raw = torch.rand(num_heads, seq_len, seq_len)
    attn = raw / raw.sum(dim=-1, keepdim=True)
    return key, val, attn


class TestKVShrinkerBasic:
    @pytest.mark.parametrize("policy", [
        EvictionPolicy.H2O,
        EvictionPolicy.SNAPKV,
        EvictionPolicy.RAG_AWARE,
        EvictionPolicy.STREAMING,
        EvictionPolicy.PYRAMID,
    ])
    def test_all_policies_run(self, policy):
        config = KVShrinkerConfig(budget_ratio=0.3, policy=policy)
        shrinker = KVShrinker(config, layer_idx=0, total_layers=32)
        key, val, attn = make_tensors()
        pk, pv, mask = shrinker.compress(key, val, attn)
        assert pk.shape[0] == mask.sum().item()
        assert pv.shape[0] == mask.sum().item()

    def test_output_shapes_consistent(self):
        config = KVShrinkerConfig(budget_ratio=0.4)
        shrinker = KVShrinker(config, layer_idx=0, total_layers=32)
        key, val, attn = make_tensors(seq_len=200)
        pk, pv, mask = shrinker.compress(key, val, attn)
        kept = mask.sum().item()
        assert pk.shape == (kept, 4, 64)
        assert pv.shape == (kept, 4, 64)
        assert mask.shape == (200,)
        assert mask.dtype == torch.bool

    def test_budget_ratio_100pct(self):
        config = KVShrinkerConfig(budget_ratio=1.0)
        shrinker = KVShrinker(config)
        key, val, attn = make_tensors(seq_len=64)
        pk, pv, mask = shrinker.compress(key, val, attn)
        assert mask.all()

    def test_sink_tokens_always_kept(self):
        config = KVShrinkerConfig(budget_ratio=0.2, sink_size=8)
        shrinker = KVShrinker(config)
        key, val, attn = make_tensors(seq_len=256)
        _, _, mask = shrinker.compress(key, val, attn)
        assert mask[:8].all(), "Sink tokens must always be kept"

    def test_window_tokens_always_kept(self):
        config = KVShrinkerConfig(budget_ratio=0.2, window_size=16)
        shrinker = KVShrinker(config)
        key, val, attn = make_tensors(seq_len=256)
        _, _, mask = shrinker.compress(key, val, attn)
        assert mask[-16:].all(), "Window tokens must always be kept"

    def test_min_budget_respected(self):
        config = KVShrinkerConfig(budget_ratio=0.01, min_budget_tokens=32)
        shrinker = KVShrinker(config)
        key, val, attn = make_tensors(seq_len=512)
        _, _, mask = shrinker.compress(key, val, attn)
        assert mask.sum().item() >= 32

    def test_reset_state(self):
        config = KVShrinkerConfig(policy=EvictionPolicy.H2O)
        shrinker = KVShrinker(config)
        key, val, attn = make_tensors()
        shrinker.compress(key, val, attn)
        assert shrinker._accumulated_scores is not None
        shrinker.reset_state()
        assert shrinker._accumulated_scores is None

    def test_repr(self):
        config = KVShrinkerConfig(budget_ratio=0.3, policy="rag_aware")
        shrinker = KVShrinker(config, layer_idx=5, total_layers=32)
        r = repr(shrinker)
        assert "rag_aware" in r
        assert "30%" in r


class TestKVShrinkerRAG:
    def test_rag_signal_boosts_keyword_tokens(self):
        """Tokens with RAG boost should be more likely to survive eviction."""
        config = KVShrinkerConfig(
            budget_ratio=0.3,
            policy=EvictionPolicy.RAG_AWARE,
            rag_alpha=10.0,  # strong RAG signal
            sink_size=0,     # no forced sinks so all budget is score-driven
            window_size=0,   # no forced window so RAG tokens compete freely
        )
        shrinker = KVShrinker(config)
        key, val, attn = make_tensors(seq_len=128)

        # Mark tokens 60..69 as highly important via RAG
        rag = RAGSignal.from_token_boosts(
            token_boosts={i: 1.0 for i in range(60, 70)}
        )
        _, _, mask_with_rag = shrinker.compress(key, val, attn, rag_signal=rag)

        # Without RAG
        shrinker.reset_state()
        _, _, mask_no_rag = shrinker.compress(key, val, attn, rag_signal=None)

        # RAG-boosted tokens should be retained
        assert mask_with_rag[60:70].all(), (
            "All RAG keyword tokens should be kept when alpha=10"
        )

    def test_no_rag_signal_is_fine(self):
        config = KVShrinkerConfig(policy=EvictionPolicy.RAG_AWARE)
        shrinker = KVShrinker(config)
        key, val, attn = make_tensors()
        pk, pv, mask = shrinker.compress(key, val, attn, rag_signal=None)
        assert pk.shape[1] == 4  # num_kv_heads preserved


class TestKVShrinkerPyramid:
    def test_upper_layers_get_smaller_budget(self):
        config = KVShrinkerConfig(budget_ratio=0.5, policy=EvictionPolicy.PYRAMID)
        key, val, attn = make_tensors(seq_len=128)

        shrinker_low = KVShrinker(config, layer_idx=0, total_layers=32)
        shrinker_high = KVShrinker(config, layer_idx=31, total_layers=32)

        _, _, mask_low = shrinker_low.compress(key, val, attn)
        _, _, mask_high = shrinker_high.compress(key, val, attn)

        assert mask_low.sum() >= mask_high.sum(), (
            "Lower layers should keep more tokens under pyramid schedule"
        )
