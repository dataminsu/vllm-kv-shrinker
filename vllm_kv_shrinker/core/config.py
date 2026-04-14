"""Configuration dataclasses for KVShrinker."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EvictionPolicy(str, Enum):
    """Supported KV cache eviction policies."""

    H2O = "h2o"                   # Heavy-Hitter Oracle (NeurIPS 2023)
    SNAPKV = "snapkv"             # Query-aware eviction (NeurIPS 2024)
    RAG_AWARE = "rag_aware"       # RAG-signal-guided (this work)
    STREAMING = "streaming"       # Attention sink + recent window
    PYRAMID = "pyramid"           # Layer-wise pyramid budget


@dataclass
class KVShrinkerConfig:
    """
    Configuration for KVShrinker.

    Args:
        budget_ratio: Fraction of KV tokens to retain (0 < ratio <= 1.0).
            e.g. 0.3 means keep 30% of tokens.
        policy: Which eviction policy to use.
        rag_alpha: Weight for the RAG lexical signal in importance scoring.
            Final score = attn_score + rag_alpha * rag_lexical + rag_beta * rag_dense
        rag_beta: Weight for the RAG dense (embedding) signal.
        sink_size: Number of initial tokens to always preserve (attention sinks).
        window_size: Number of recent tokens to always preserve.
        obs_window_size: For SnapKV-style scoring, the observation window length
            (number of instruction/query tokens whose attention is averaged).
        layer_budget_schedule: If "pyramid", fraction of budget per layer index.
            Dict mapping layer_idx -> budget_ratio override.
        min_budget_tokens: Hard lower bound on number of tokens kept,
            regardless of budget_ratio.
        device: Torch device for internal computations.
    """

    budget_ratio: float = 0.3
    policy: EvictionPolicy = EvictionPolicy.RAG_AWARE
    rag_alpha: float = 1.0
    rag_beta: float = 0.5
    sink_size: int = 4
    window_size: int = 32
    obs_window_size: int = 16
    layer_budget_schedule: Optional[dict] = None
    min_budget_tokens: int = 16
    device: str = "cuda"

    def __post_init__(self):
        if isinstance(self.policy, str):
            self.policy = EvictionPolicy(self.policy)
        assert 0 < self.budget_ratio <= 1.0, "budget_ratio must be in (0, 1]"
        assert self.sink_size >= 0
        assert self.window_size >= 0

    def get_layer_budget(self, layer_idx: int, total_layers: int) -> float:
        """Return budget ratio for a given layer (supports pyramid schedule)."""
        if self.layer_budget_schedule and layer_idx in self.layer_budget_schedule:
            return self.layer_budget_schedule[layer_idx]
        if self.policy == EvictionPolicy.PYRAMID:
            # Upper layers get progressively tighter budgets.
            # Layer 0 -> budget_ratio, last layer -> budget_ratio / 2
            decay = 1.0 - 0.5 * (layer_idx / max(total_layers - 1, 1))
            return max(self.budget_ratio * decay, self.min_budget_tokens / 128)
        return self.budget_ratio
