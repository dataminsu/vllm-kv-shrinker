"""Tests for importance scorers."""

import pytest
import torch

from vllm_kv_shrinker.core.config import EvictionPolicy, KVShrinkerConfig
from vllm_kv_shrinker.core.importance_scorer import (
    H2OScorer,
    RAGAwareScorer,
    SnapKVScorer,
    build_scorer,
)


@pytest.fixture
def attn_weights():
    """Synthetic [num_heads, q_len, k_len] attention weights."""
    torch.manual_seed(42)
    w = torch.rand(8, 16, 64)
    return w / w.sum(dim=-1, keepdim=True)  # normalize


class TestH2OScorer:
    def test_output_shape(self, attn_weights):
        scorer = H2OScorer()
        scores = scorer.score(attn_weights)
        assert scores.shape == (64,)

    def test_accumulation(self, attn_weights):
        scorer = H2OScorer()
        s1 = scorer.score(attn_weights)
        s2 = scorer.score(attn_weights, accumulated=s1)
        # Second call should be strictly larger (accumulated > 0)
        assert (s2 > s1).all()

    def test_decay(self, attn_weights):
        scorer_decay = H2OScorer(decay=0.5)
        scorer_no_decay = H2OScorer(decay=1.0)
        s1 = scorer_no_decay.score(attn_weights)
        s2 = scorer_no_decay.score(attn_weights, accumulated=s1)
        s1d = scorer_decay.score(attn_weights)
        s2d = scorer_decay.score(attn_weights, accumulated=s1d)
        # With decay the scores grow more slowly
        assert (s2d <= s2).all()

    def test_scores_nonnegative(self, attn_weights):
        scorer = H2OScorer()
        scores = scorer.score(attn_weights)
        assert (scores >= 0).all()


class TestSnapKVScorer:
    def test_output_shape(self, attn_weights):
        scorer = SnapKVScorer(obs_window=16)
        scores = scorer.score(attn_weights)
        assert scores.shape == (64,)

    def test_obs_window_smaller_than_q_len(self, attn_weights):
        scorer = SnapKVScorer(obs_window=4)
        scores = scorer.score(attn_weights)
        assert scores.shape == (64,)

    def test_obs_window_larger_than_q_len(self):
        # obs_window > q_len: should clamp gracefully
        w = torch.rand(4, 8, 32)
        w = w / w.sum(dim=-1, keepdim=True)
        scorer = SnapKVScorer(obs_window=100)
        scores = scorer.score(w)
        assert scores.shape == (32,)

    def test_no_pooling(self, attn_weights):
        scorer = SnapKVScorer(obs_window=16, pooling_kernel=1)
        scores = scorer.score(attn_weights)
        assert scores.shape == (64,)

    def test_scores_nonnegative(self, attn_weights):
        scorer = SnapKVScorer()
        scores = scorer.score(attn_weights)
        assert (scores >= 0).all()


class TestRAGAwareScorer:
    def test_no_rag_signal(self, attn_weights):
        scorer = RAGAwareScorer()
        scores = scorer.score(attn_weights)
        assert scores.shape == (64,)

    def test_with_lexical_boost(self, attn_weights):
        scorer = RAGAwareScorer(alpha=2.0, beta=0.0)
        base_scores = scorer.score(attn_weights)

        # Create a boost that marks tokens 10..14 as important
        boost = torch.zeros(64)
        boost[10:15] = 1.0
        rag_scores = scorer.score(attn_weights, rag_boost=boost)

        # Boosted tokens should have higher scores
        assert (rag_scores[10:15] > base_scores[10:15]).all()
        # Unboosted tokens should be unchanged
        assert torch.allclose(rag_scores[:10], base_scores[:10])

    def test_with_dense_boost(self, attn_weights):
        scorer = RAGAwareScorer(alpha=0.0, beta=1.0)
        dense = torch.zeros(64)
        dense[20:25] = 0.9
        scores = scorer.score(attn_weights, rag_dense_boost=dense)
        assert scores.shape == (64,)
        # Dense-boosted tokens must be among highest
        top5 = scores.topk(5).indices.tolist()
        assert any(20 <= i < 25 for i in top5)

    def test_boost_length_mismatch(self, attn_weights):
        # boost tensor shorter than k_len: should not raise
        scorer = RAGAwareScorer()
        boost = torch.ones(32)  # k_len=64 but boost has 32
        scores = scorer.score(attn_weights, rag_boost=boost)
        assert scores.shape == (64,)

    def test_boost_length_longer(self, attn_weights):
        # boost tensor longer than k_len: should truncate
        scorer = RAGAwareScorer()
        boost = torch.ones(100)
        scores = scorer.score(attn_weights, rag_boost=boost)
        assert scores.shape == (64,)


class TestBuildScorer:
    @pytest.mark.parametrize("policy", [
        EvictionPolicy.H2O,
        EvictionPolicy.SNAPKV,
        EvictionPolicy.RAG_AWARE,
        EvictionPolicy.STREAMING,
        EvictionPolicy.PYRAMID,
    ])
    def test_build_all_policies(self, policy):
        config = KVShrinkerConfig(policy=policy)
        scorer = build_scorer(policy.value, config)
        assert scorer is not None
