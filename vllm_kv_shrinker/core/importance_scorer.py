"""
Token importance scorers for KV cache pruning.

Each scorer takes attention weights and optional signals, and returns
a per-token importance score tensor of shape [seq_len].

Implemented methods:
  - H2OScorer      : cumulative attention sum (NeurIPS 2023)
  - SnapKVScorer   : observation-window pooling (NeurIPS 2024)
  - RAGAwareScorer : attention + external RAG signal (this work)
"""

from __future__ import annotations

import abc
from typing import Optional

import torch
import torch.nn.functional as F


class ImportanceScorer(abc.ABC):
    """Abstract base class for token importance scorers."""

    @abc.abstractmethod
    def score(
        self,
        attn_weights: torch.Tensor,
        rag_boost: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute per-token importance scores.

        Args:
            attn_weights: Attention weight tensor.
                Shape: [num_heads, query_len, key_len]  (prefill)
                    or [num_heads, 1, key_len]           (decode step)
            rag_boost: Optional per-token RAG importance boost.
                Shape: [key_len]  (values in [0, 1], 0 = no boost)
            **kwargs: Scorer-specific extra arguments.

        Returns:
            scores: Per-token importance score. Shape: [key_len].
                Higher = more important = should be kept.
        """


# ---------------------------------------------------------------------------
# H2O Scorer (Zhang et al., NeurIPS 2023)
# ---------------------------------------------------------------------------

class H2OScorer(ImportanceScorer):
    """
    Heavy-Hitter Oracle scorer.

    Importance = cumulative sum of attention weights across all heads and
    all query positions seen so far.  Tokens that consistently receive
    high attention are 'heavy hitters' and should be retained.

    Reference: H2O: Heavy-Hitter Oracle for Efficient Generative Inference
               of Large Language Models (NeurIPS 2023).
    """

    def __init__(self, decay: float = 1.0):
        """
        Args:
            decay: Exponential decay applied to the accumulated score at
                   each new decode step (1.0 = no decay, H2O original).
        """
        self.decay = decay

    def score(
        self,
        attn_weights: torch.Tensor,
        rag_boost: Optional[torch.Tensor] = None,
        accumulated: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            attn_weights: [num_heads, q_len, k_len]
            rag_boost:    [k_len]  (unused by H2O; kept for interface compat.)
            accumulated:  [k_len]  running importance from prior steps.

        Returns:
            scores: [k_len]
        """
        # Average over heads, sum over query positions
        # -> [k_len]
        step_score = attn_weights.mean(dim=0).sum(dim=0)

        if accumulated is not None:
            return self.decay * accumulated + step_score
        return step_score


# ---------------------------------------------------------------------------
# SnapKV Scorer (Li et al., NeurIPS 2024)
# ---------------------------------------------------------------------------

class SnapKVScorer(ImportanceScorer):
    """
    Query-aware scorer that uses the attention pattern of the instruction
    (observation) window to identify which prefix tokens are important
    *before* generation begins.

    The key insight: aggregate attention from the last `obs_window` query
    tokens (the instruction/query portion of the prompt) to decide which
    KV cache entries to keep.

    Reference: SnapKV: LLM Knows What You are Looking for Before Generation
               (NeurIPS 2024).
    """

    def __init__(self, obs_window: int = 16, pooling_kernel: int = 5):
        """
        Args:
            obs_window:     Number of trailing query tokens used as
                            the observation window.
            pooling_kernel: Local max-pooling kernel size for context
                            preservation around important tokens.
        """
        self.obs_window = obs_window
        self.pooling_kernel = pooling_kernel

    def score(
        self,
        attn_weights: torch.Tensor,
        rag_boost: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            attn_weights: [num_heads, q_len, k_len]
            rag_boost:    [k_len]  (unused by vanilla SnapKV)

        Returns:
            scores: [k_len]
        """
        q_len = attn_weights.shape[1]
        obs = min(self.obs_window, q_len)

        # Use only the last `obs` query tokens
        obs_attn = attn_weights[:, -obs:, :]   # [heads, obs, k_len]

        # Mean over heads and observation queries -> [k_len]
        scores = obs_attn.mean(dim=0).mean(dim=0)

        # Local max-pool to preserve context around important tokens
        if self.pooling_kernel > 1:
            pad = self.pooling_kernel // 2
            scores = F.max_pool1d(
                scores.unsqueeze(0).unsqueeze(0),
                kernel_size=self.pooling_kernel,
                stride=1,
                padding=pad,
            ).squeeze()

        return scores


# ---------------------------------------------------------------------------
# RAG-Aware Scorer (this work)
# ---------------------------------------------------------------------------

class RAGAwareScorer(ImportanceScorer):
    """
    Retrieval-Augmented KV importance scorer.

    Combines internal attention-based importance with an external RAG
    retrieval signal (e.g. BM25 term weights or dense similarity scores)
    to produce a hybrid importance score.

    Score formula:
        RTIS(t) = attn_score(t)
                  + alpha * lexical_boost(t)
                  + beta  * dense_boost(t)

    Where:
        attn_score    = SnapKV-style observation-window attention mean
        lexical_boost = BM25/TF-IDF term importance for token t
        dense_boost   = dense retriever cosine similarity for token t

    Tokens matching RAG keywords are protected from eviction even when
    their raw attention score is low — capturing "semantically important
    but attention-sparse" tokens that attention-only methods miss.

    Reference: This work (RAG-KV).
    """

    def __init__(
        self,
        obs_window: int = 16,
        pooling_kernel: int = 5,
        alpha: float = 1.0,
        beta: float = 0.5,
    ):
        """
        Args:
            obs_window:     SnapKV-style observation window size.
            pooling_kernel: Local pooling kernel for context preservation.
            alpha:          Weight for lexical RAG boost.
            beta:           Weight for dense RAG boost.
        """
        self.obs_window = obs_window
        self.pooling_kernel = pooling_kernel
        self.alpha = alpha
        self.beta = beta
        self._base_scorer = SnapKVScorer(obs_window, pooling_kernel)

    def score(
        self,
        attn_weights: torch.Tensor,
        rag_boost: Optional[torch.Tensor] = None,
        rag_dense_boost: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            attn_weights:    [num_heads, q_len, k_len]
            rag_boost:       [k_len]  lexical RAG signal (BM25/TF-IDF).
                             Values in [0, 1].  None → no lexical signal.
            rag_dense_boost: [k_len]  dense RAG signal (embedding sim).
                             Values in [0, 1].  None → no dense signal.

        Returns:
            scores: [k_len]  Retrieval-augmented token importance scores.
        """
        # Base attention score
        scores = self._base_scorer.score(attn_weights)

        k_len = scores.shape[0]

        # Add lexical RAG boost
        if rag_boost is not None:
            boost = rag_boost.to(scores.device)
            if boost.shape[0] != k_len:
                # Truncate or pad to match current sequence length
                boost = _align_tensor(boost, k_len, scores.device)
            scores = scores + self.alpha * boost

        # Add dense RAG boost
        if rag_dense_boost is not None:
            dense = rag_dense_boost.to(scores.device)
            if dense.shape[0] != k_len:
                dense = _align_tensor(dense, k_len, scores.device)
            scores = scores + self.beta * dense

        return scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _align_tensor(t: torch.Tensor, target_len: int, device: torch.device) -> torch.Tensor:
    """Truncate or zero-pad tensor to target_len along dim 0."""
    cur = t.shape[0]
    if cur >= target_len:
        return t[:target_len]
    pad = torch.zeros(target_len - cur, dtype=t.dtype, device=device)
    return torch.cat([t, pad], dim=0)


def build_scorer(policy: str, config) -> ImportanceScorer:
    """Factory: create a scorer from a policy name and KVShrinkerConfig."""
    from vllm_kv_shrinker.core.config import EvictionPolicy

    p = EvictionPolicy(policy)
    if p == EvictionPolicy.H2O:
        return H2OScorer()
    if p in (EvictionPolicy.SNAPKV, EvictionPolicy.STREAMING, EvictionPolicy.PYRAMID):
        return SnapKVScorer(obs_window=config.obs_window_size)
    if p == EvictionPolicy.RAG_AWARE:
        return RAGAwareScorer(
            obs_window=config.obs_window_size,
            alpha=config.rag_alpha,
            beta=config.rag_beta,
        )
    raise ValueError(f"Unknown policy: {policy}")
