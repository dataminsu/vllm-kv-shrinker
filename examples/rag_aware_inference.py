"""
RAG-aware KV pruning example.

Shows how a RAG pipeline passes retrieval keyword scores to KVShrinker
to protect semantically important tokens.

In production this would be integrated with vLLM's serving loop;
here we demonstrate the signal flow on synthetic data.
"""

import torch
from vllm_kv_shrinker import (
    KVShrinker,
    KVShrinkerConfig,
    EvictionPolicy,
    RAGSignal,
)

# ── Simulated RAG output ──────────────────────────────────────────────────
# In a real pipeline, BM25 retriever returns keyword importance scores.
# Normalise to [0, 1].
retrieved_keywords = {
    "PagedAttention":  1.0,
    "KV cache":        0.95,
    "vLLM":            0.88,
    "memory":          0.70,
    "throughput":      0.65,
}

# Simulated input token ids (in practice: tokenizer(prompt).input_ids)
# We'll manually plant the keyword positions for demonstration.
torch.manual_seed(7)
seq_len = 256
input_ids = torch.randint(100, 5000, (seq_len,))

# Plant keyword tokens at known positions
# "PagedAttention" → token ids [1111, 2222] at positions 40-41
input_ids[40] = 1111
input_ids[41] = 2222
# "KV cache" → token ids [3333, 4444] at positions 80-81
input_ids[80] = 3333
input_ids[81] = 4444


class FakeTokenizer:
    """Simulates tokenizer.encode() for demo purposes."""
    _map = {
        "PagedAttention": [1111, 2222],
        "KV cache": [3333, 4444],
        "vLLM": [5555],
        "memory": [6666],
        "throughput": [7777],
    }

    def encode(self, text, add_special_tokens=False):
        return self._map.get(text, [])


# ── Build RAGSignal ────────────────────────────────────────────────────────
rag = RAGSignal.from_keywords(
    keywords=retrieved_keywords,
    tokenizer=FakeTokenizer(),
    input_ids=input_ids,
)
print(f"RAG signal: {rag.num_boosted_tokens()} token(s) boosted")
print(f"Boosted positions: {rag.boosted_indices()}")

# ── KVShrinker ─────────────────────────────────────────────────────────────
config = KVShrinkerConfig(
    budget_ratio=0.25,
    policy=EvictionPolicy.RAG_AWARE,
    rag_alpha=2.0,
    rag_beta=0.5,
    sink_size=4,
    window_size=16,
)
shrinker = KVShrinker(config, layer_idx=16, total_layers=32)

# Synthetic KV / attention
num_kv_heads, head_dim, num_heads = 8, 64, 32
key   = torch.randn(seq_len, num_kv_heads, head_dim)
value = torch.randn(seq_len, num_kv_heads, head_dim)
raw   = torch.rand(num_heads, seq_len, seq_len)
attn  = raw / raw.sum(dim=-1, keepdim=True)

# ── Compress with RAG guidance ─────────────────────────────────────────────
pk, pv, mask = shrinker.compress(key, value, attn, rag_signal=rag)

print(f"\nKept {mask.sum().item()}/{seq_len} tokens ({mask.sum().item()/seq_len:.1%})")
print(f"'PagedAttention' tokens (40-41) kept: {mask[40].item()} / {mask[41].item()}")
print(f"'KV cache' tokens (80-81) kept      : {mask[80].item()} / {mask[81].item()}")

# ── Compare: same budget without RAG ──────────────────────────────────────
from vllm_kv_shrinker.core.config import EvictionPolicy as EP
config_norag = KVShrinkerConfig(budget_ratio=0.25, policy=EP.SNAPKV)
shrinker_norag = KVShrinker(config_norag, layer_idx=16, total_layers=32)
_, _, mask_norag = shrinker_norag.compress(key, value, attn)

kw_positions = [40, 41, 80, 81]
rag_retained  = sum(mask[p].item()      for p in kw_positions)
norag_retained = sum(mask_norag[p].item() for p in kw_positions)
print(f"\nKeyword retention — RAG-aware: {rag_retained}/4  |  SnapKV: {norag_retained}/4")
