"""
vLLM AttentionLayer patch for KVShrinker integration.

This module provides the minimal diff needed to wire KVShrinker into
vLLM's existing Attention class.  It can be applied in two ways:

  Option A — Monkey-patch (dev/experiment):
      from vllm_kv_shrinker.vllm_patch.attention_patch import patch_vllm_attention
      patch_vllm_attention(config)

  Option B — Upstream PR:
      Copy the patch_attention_layer() logic into
      vllm/attention/layer.py and add KVShrinkerConfig to
      vllm/config.py::CacheConfig.

Design notes
------------
vLLM's Attention.forward() signature (simplified):

    def forward(
        self,
        query:        Tensor,   # [num_tokens, num_heads * head_dim]
        key:          Tensor,   # [num_tokens, num_kv_heads * head_dim]
        value:        Tensor,   # [num_tokens, num_kv_heads * head_dim]
        kv_cache:     Tensor,
        attn_metadata: AttentionMetadata,
    ) -> Tensor:

The hook is inserted after QKV projection but before the paged attention
kernel writes into kv_cache.  We intercept key/value, run the shrinker,
and pass the pruned tensors downstream.

AttentionMetadata is extended with an optional `rag_signal` field that
the RAG layer populates before calling model.generate().
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from vllm_kv_shrinker.core.config import KVShrinkerConfig
from vllm_kv_shrinker.core.kv_shrinker import KVShrinker
from vllm_kv_shrinker.rag.rag_signal import RAGSignal

if TYPE_CHECKING:
    pass  # avoid circular imports with vllm types at module level


# ---------------------------------------------------------------------------
# AttentionMetadata extension
# ---------------------------------------------------------------------------

def extend_attn_metadata(attn_metadata, rag_signal: Optional[RAGSignal] = None):
    """
    Attach a RAGSignal to an existing AttentionMetadata object.

    In a proper PR, RAGSignal would be a first-class field on
    AttentionMetadata.  For monkey-patching, we simply set the attribute.

    Usage:
        extend_attn_metadata(model_runner.attn_metadata, rag_signal)
    """
    attn_metadata.rag_signal = rag_signal
    return attn_metadata


# ---------------------------------------------------------------------------
# Layer-level patch
# ---------------------------------------------------------------------------

def patch_attention_layer(attn_layer, config: KVShrinkerConfig, layer_idx: int, total_layers: int):
    """
    Inject a KVShrinker into a single vLLM Attention layer instance.

    Args:
        attn_layer:   A vllm.attention.layer.Attention instance.
        config:       KVShrinkerConfig.
        layer_idx:    Zero-based index of this layer.
        total_layers: Total number of transformer layers.
    """
    shrinker = KVShrinker(config, layer_idx=layer_idx, total_layers=total_layers)

    # Store shrinker on the layer
    attn_layer._kv_shrinker = shrinker

    # Wrap the forward method
    original_forward = attn_layer.forward

    def patched_forward(query, key, value, kv_cache, attn_metadata, *args, **kwargs):
        # ---------------------------------------------------------------
        # 1. Reshape key/value to [seq_len, num_kv_heads, head_dim]
        #    (vLLM packs heads into last dim, so we need to reshape)
        # ---------------------------------------------------------------
        num_tokens = key.shape[0]
        num_kv_heads = attn_layer.num_kv_heads
        head_dim = key.shape[-1] // num_kv_heads

        key_3d = key.view(num_tokens, num_kv_heads, head_dim)
        val_3d = value.view(num_tokens, num_kv_heads, head_dim)

        # ---------------------------------------------------------------
        # 2. Build a proxy attention weight tensor for importance scoring.
        #    During prefill we have access to full QK^T; during decode
        #    we use a simplified dot-product proxy.
        # ---------------------------------------------------------------
        num_heads = attn_layer.num_heads
        q_3d = query.view(num_tokens, num_heads, head_dim)
        # Proxy: [num_heads, num_tokens, num_tokens] (cheap approximation)
        # For large seqs this is expensive — a real PR would hook into
        # the Flash Attention kernel to get attention weights for free.
        proxy_attn = _compute_proxy_attn(q_3d, key_3d, num_heads, num_kv_heads)

        # ---------------------------------------------------------------
        # 3. Get RAG signal if available
        # ---------------------------------------------------------------
        rag_signal = getattr(attn_metadata, "rag_signal", None)

        # ---------------------------------------------------------------
        # 4. Run KVShrinker
        # ---------------------------------------------------------------
        pruned_key, pruned_val, keep_mask = attn_layer._kv_shrinker.compress(
            key_3d, val_3d, proxy_attn, rag_signal=rag_signal
        )

        # Reshape back to vLLM's expected [kept_tokens, num_kv_heads * head_dim]
        kept = pruned_key.shape[0]
        pruned_key_flat = pruned_key.view(kept, num_kv_heads * head_dim)
        pruned_val_flat = pruned_val.view(kept, num_kv_heads * head_dim)

        # ---------------------------------------------------------------
        # 5. Continue with original forward on pruned KV
        # ---------------------------------------------------------------
        # NOTE: In a real PR, the original forward would accept the pruned
        # key/value directly. Here we pass them as replacements.
        return original_forward(
            query, pruned_key_flat, pruned_val_flat, kv_cache, attn_metadata,
            *args, **kwargs
        )

    attn_layer.forward = patched_forward
    return attn_layer


def _compute_proxy_attn(
    query: torch.Tensor,   # [seq_len, num_heads, head_dim]
    key: torch.Tensor,     # [seq_len, num_kv_heads, head_dim]
    num_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """
    Compute a cheap proxy attention weight matrix.
    Returns shape [num_heads, seq_len, seq_len].

    For GQA models (num_heads > num_kv_heads), keys are repeated to match
    the query head count.
    """
    seq_len, _, head_dim = query.shape

    # Handle GQA: repeat keys/values
    if num_heads != num_kv_heads:
        repeat = num_heads // num_kv_heads
        key = key.repeat_interleave(repeat, dim=1)  # [seq_len, num_heads, head_dim]

    # [num_heads, seq_len, head_dim]
    q = query.permute(1, 0, 2)
    k = key.permute(1, 0, 2)

    # [num_heads, seq_len, seq_len]
    scale = head_dim ** -0.5
    attn = torch.bmm(q, k.transpose(1, 2)) * scale
    attn = torch.softmax(attn, dim=-1)
    return attn


# ---------------------------------------------------------------------------
# Model-level patch (patches all Attention layers in a model)
# ---------------------------------------------------------------------------

def patch_vllm_attention(
    model: torch.nn.Module,
    config: KVShrinkerConfig,
    attn_class_name: str = "Attention",
) -> int:
    """
    Walk all modules in `model` and inject KVShrinker into every
    instance of the attention class.

    Args:
        model:           vLLM LLM model (e.g. LlamaModel).
        config:          KVShrinkerConfig to use for all layers.
        attn_class_name: Class name of the attention module to patch.

    Returns:
        Number of layers patched.
    """
    attention_layers = [
        (name, module)
        for name, module in model.named_modules()
        if type(module).__name__ == attn_class_name
    ]
    total = len(attention_layers)

    for idx, (name, layer) in enumerate(attention_layers):
        patch_attention_layer(layer, config, layer_idx=idx, total_layers=total)

    return total
