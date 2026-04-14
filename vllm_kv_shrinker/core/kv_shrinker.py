"""
KVShrinker: main orchestrator for KV cache compression.

This is the primary class that integrates with vLLM's attention layer.
It wraps around importance scoring + eviction policy and exposes a
clean `compress()` interface that vLLM's AttentionLayer can call.

Integration point in vLLM (vllm/attention/layer.py):

    class Attention(nn.Module):
        def __init__(self, ...):
            ...
            if kv_shrinker_config is not None:
                self.kv_shrinker = KVShrinker(kv_shrinker_config, layer_idx)

        def forward(self, query, key, value, kv_cache, attn_metadata):
            ...
            # After computing attention weights (pre-softmax or post-softmax):
            if hasattr(self, 'kv_shrinker'):
                key, value = self.kv_shrinker.compress(
                    key, value, attn_weights,
                    rag_signal=attn_metadata.rag_signal,
                )
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from vllm_kv_shrinker.core.config import EvictionPolicy, KVShrinkerConfig
from vllm_kv_shrinker.core.eviction_policy import apply_eviction, apply_eviction_to_kv
from vllm_kv_shrinker.core.importance_scorer import (
    ImportanceScorer,
    build_scorer,
)
from vllm_kv_shrinker.core.kv_quantizer import KVQuantizer, QuantizedKV
from vllm_kv_shrinker.rag.rag_signal import RAGSignal


class KVShrinker:
    """
    KV Cache Shrinker — drop-in token eviction layer for vLLM.

    Supports multiple eviction policies (H2O, SnapKV, RAG-aware, Pyramid)
    via a unified interface.  The RAG-aware policy uses external retrieval
    signals to protect semantically important tokens even when their
    attention scores are low.

    Example usage (standalone, no vLLM):
        config = KVShrinkerConfig(budget_ratio=0.3, policy="rag_aware")
        shrinker = KVShrinker(config, layer_idx=0, total_layers=32)

        rag = RAGSignal.from_keywords({"Paris": 1.0, "capital": 0.8}, tokenizer)
        k_pruned, v_pruned, mask = shrinker.compress(
            key, value, attn_weights, rag_signal=rag
        )
    """

    def __init__(
        self,
        config: KVShrinkerConfig,
        layer_idx: int = 0,
        total_layers: int = 32,
    ):
        """
        Args:
            config:       KVShrinkerConfig instance.
            layer_idx:    Index of this attention layer (0-based).
                          Used for pyramid budget scheduling.
            total_layers: Total number of attention layers in the model.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.total_layers = total_layers

        self.scorer: ImportanceScorer = build_scorer(config.policy, config)

        # For H2O: maintain running accumulated scores across decode steps
        self._accumulated_scores: Optional[torch.Tensor] = None

        # Optional post-eviction quantizer
        self.quantizer: Optional[KVQuantizer] = (
            KVQuantizer(bits=config.quant_bits, group_size=config.quant_group_size)
            if config.quant_bits is not None
            else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_weights: torch.Tensor,
        rag_signal: Optional[RAGSignal] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compress the KV cache by evicting low-importance tokens.

        Args:
            key:          [seq_len, num_kv_heads, head_dim]
            value:        [seq_len, num_kv_heads, head_dim]
            attn_weights: [num_heads, q_len, seq_len]
                          Attention weights (post-softmax preferred;
                          pre-softmax logits also accepted).
            rag_signal:   Optional RAGSignal carrying per-token importance
                          boosts derived from retrieval keywords.

        Returns:
            pruned_key:   [kept, num_kv_heads, head_dim]
            pruned_value: [kept, num_kv_heads, head_dim]
            keep_mask:    [seq_len] bool — True for kept tokens.
        """
        seq_len = key.shape[0]

        # Trivial path: nothing to compress
        budget_ratio = self.config.get_layer_budget(self.layer_idx, self.total_layers)
        budget = max(int(seq_len * budget_ratio), self.config.min_budget_tokens)
        if budget >= seq_len:
            keep_mask = torch.ones(seq_len, dtype=torch.bool, device=key.device)
            return key, value, keep_mask

        # --- Compute importance scores ---
        rag_lexical = rag_dense = None
        if rag_signal is not None:
            rag_lexical, rag_dense = rag_signal.get_token_boosts(seq_len, key.device)

        if self.config.policy == EvictionPolicy.H2O:
            scores = self.scorer.score(
                attn_weights,
                accumulated=self._accumulated_scores,
            )
            self._accumulated_scores = scores.detach()
        else:
            scores = self.scorer.score(
                attn_weights,
                rag_boost=rag_lexical,
                rag_dense_boost=rag_dense,
            )

        # --- Apply eviction policy ---
        keep_mask = apply_eviction(
            scores,
            budget_ratio=budget_ratio,
            sink_size=self.config.sink_size,
            window_size=self.config.window_size,
            min_budget=self.config.min_budget_tokens,
        )

        # --- Filter KV tensors ---
        pruned_key, pruned_value, _ = apply_eviction_to_kv(key, value, keep_mask)

        # Optional post-eviction quantization (quant → dequant to preserve dtype)
        if self.quantizer is not None:
            pruned_key = self.quantizer.quantize_and_dequantize(pruned_key)
            pruned_value = self.quantizer.quantize_and_dequantize(pruned_value)

        return pruned_key, pruned_value, keep_mask

    def compress_quantized(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_weights: torch.Tensor,
        rag_signal: Optional[RAGSignal] = None,
    ) -> Tuple["QuantizedKV", "QuantizedKV", torch.Tensor]:
        """
        Like compress(), but returns QuantizedKV objects instead of float tensors.

        Requires quant_bits to be set in config.  Caller is responsible for
        calling .dequantize() before passing to attention kernels.

        Returns:
            q_key:    QuantizedKV for keys
            q_value:  QuantizedKV for values
            keep_mask: [seq_len] bool
        """
        if self.quantizer is None:
            raise RuntimeError(
                "compress_quantized() requires quant_bits to be set in KVShrinkerConfig"
            )
        pruned_key, pruned_value, keep_mask = self.compress(
            key, value, attn_weights, rag_signal
        )
        # compress() already applied quant-dequant; redo quantize-only for the
        # quantized output path — temporarily bypass the quant-dequant in compress()
        # by quantizing the pruned float tensors directly.
        q_key = self.quantizer.quantize(pruned_key)
        q_value = self.quantizer.quantize(pruned_value)
        return q_key, q_value, keep_mask

    def reset_state(self) -> None:
        """Reset per-sequence accumulated state (call between requests)."""
        self._accumulated_scores = None

    @property
    def compression_ratio(self) -> float:
        """Nominal compression ratio (1 / budget_ratio)."""
        return 1.0 / self.config.budget_ratio

    def __repr__(self) -> str:
        return (
            f"KVShrinker(policy={self.config.policy.value}, "
            f"budget={self.config.budget_ratio:.0%}, "
            f"layer={self.layer_idx}/{self.total_layers}, "
            f"sink={self.config.sink_size}, window={self.config.window_size})"
        )
