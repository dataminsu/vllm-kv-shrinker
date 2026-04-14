"""
vllm-kv-shrinker: RAG-aware KV Cache Pruning for vLLM
======================================================

A drop-in KV cache compression layer for vLLM that uses
retrieval-augmented signals to preserve semantically important tokens.

Quickstart:
    from vllm_kv_shrinker import KVShrinker, KVShrinkerConfig, RAGSignal

    config = KVShrinkerConfig(budget_ratio=0.3, policy="rag_aware")
    shrinker = KVShrinker(config)
"""

from vllm_kv_shrinker.core.kv_shrinker import KVShrinker
from vllm_kv_shrinker.core.config import KVShrinkerConfig, EvictionPolicy
from vllm_kv_shrinker.rag.rag_signal import RAGSignal, RAGKeywordScore
from vllm_kv_shrinker.core.importance_scorer import (
    ImportanceScorer,
    H2OScorer,
    SnapKVScorer,
    RAGAwareScorer,
)

__all__ = [
    "KVShrinker",
    "KVShrinkerConfig",
    "EvictionPolicy",
    "RAGSignal",
    "RAGKeywordScore",
    "ImportanceScorer",
    "H2OScorer",
    "SnapKVScorer",
    "RAGAwareScorer",
]

__version__ = "0.1.0"
