"""Tests for eviction policy and KV filtering."""

import pytest
import torch

from vllm_kv_shrinker.core.eviction_policy import (
    apply_eviction,
    apply_eviction_to_kv,
    select_tokens_to_keep,
)


@pytest.fixture
def scores():
    torch.manual_seed(0)
    return torch.rand(128)


class TestSelectTokensToKeep:
    def test_budget_respected(self, scores):
        mask = select_tokens_to_keep(scores, budget=40, sink_size=4, window_size=8)
        assert mask.sum().item() <= 40 + 8  # at most budget + window overlap

    def test_sink_always_kept(self, scores):
        mask = select_tokens_to_keep(scores, budget=20, sink_size=4, window_size=4)
        assert mask[:4].all()

    def test_window_always_kept(self, scores):
        mask = select_tokens_to_keep(scores, budget=20, sink_size=4, window_size=8)
        assert mask[-8:].all()

    def test_budget_ge_seq_len(self, scores):
        # budget >= seq_len: keep everything
        mask = select_tokens_to_keep(scores, budget=200, sink_size=4, window_size=8)
        assert mask.all()

    def test_min_budget_enforced(self):
        s = torch.rand(64)
        mask = select_tokens_to_keep(s, budget=2, sink_size=0, window_size=0, min_budget=16)
        assert mask.sum().item() >= 16

    def test_top_tokens_kept(self):
        # Make token 50 the highest scorer; it should be kept
        s = torch.zeros(100)
        s[50] = 10.0
        mask = select_tokens_to_keep(s, budget=10, sink_size=4, window_size=4)
        assert mask[50].item()

    def test_output_dtype(self, scores):
        mask = select_tokens_to_keep(scores, budget=30)
        assert mask.dtype == torch.bool


class TestApplyEviction:
    def test_ratio_30pct(self, scores):
        mask = apply_eviction(scores, budget_ratio=0.3, sink_size=4, window_size=8)
        assert mask.dtype == torch.bool
        # At least 30% kept (sink/window may push it higher)
        assert mask.sum().item() >= int(128 * 0.3)

    def test_ratio_100pct(self, scores):
        mask = apply_eviction(scores, budget_ratio=1.0)
        assert mask.all()

    def test_ratio_very_small(self, scores):
        mask = apply_eviction(scores, budget_ratio=0.01, min_budget=8)
        assert mask.sum().item() >= 8


class TestApplyEvictionToKV:
    def test_shapes(self):
        seq_len, nheads, hdim = 64, 8, 128
        key = torch.randn(seq_len, nheads, hdim)
        val = torch.randn(seq_len, nheads, hdim)
        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[:20] = True

        pk, pv, indices = apply_eviction_to_kv(key, val, mask)
        assert pk.shape == (20, nheads, hdim)
        assert pv.shape == (20, nheads, hdim)
        assert indices.shape == (20,)

    def test_content_preserved(self):
        seq_len = 32
        key = torch.arange(seq_len).float().unsqueeze(-1).unsqueeze(-1).expand(-1, 2, 4)
        val = key.clone()
        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[5] = True
        mask[15] = True

        pk, pv, indices = apply_eviction_to_kv(key, val, mask)
        assert indices.tolist() == [5, 15]
        assert pk[0, 0, 0].item() == 5.0
        assert pk[1, 0, 0].item() == 15.0
