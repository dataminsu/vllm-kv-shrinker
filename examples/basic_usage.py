"""
Basic KVShrinker usage — no vLLM required.

Demonstrates the core API on synthetic tensors.
"""

import torch
from vllm_kv_shrinker import KVShrinker, KVShrinkerConfig, EvictionPolicy

# ── 1. Create a shrinker with 30% budget ──────────────────────────────────
config = KVShrinkerConfig(
    budget_ratio=0.3,
    policy=EvictionPolicy.SNAPKV,
    sink_size=4,
    window_size=16,
)
shrinker = KVShrinker(config, layer_idx=12, total_layers=32)
print(shrinker)

# ── 2. Synthetic KV tensors (as vLLM would provide) ───────────────────────
seq_len, num_kv_heads, head_dim = 512, 8, 128
num_heads = 32

key   = torch.randn(seq_len, num_kv_heads, head_dim)
value = torch.randn(seq_len, num_kv_heads, head_dim)

# Attention weights [num_heads, q_len, k_len]
raw = torch.rand(num_heads, seq_len, seq_len)
attn_weights = raw / raw.sum(dim=-1, keepdim=True)

# ── 3. Compress ────────────────────────────────────────────────────────────
pruned_key, pruned_val, keep_mask = shrinker.compress(key, value, attn_weights)

kept = keep_mask.sum().item()
print(f"Kept {kept}/{seq_len} tokens ({kept/seq_len:.1%})")
print(f"Pruned key shape : {pruned_key.shape}")
print(f"Pruned val shape : {pruned_val.shape}")
print(f"Sink tokens kept : {keep_mask[:4].all().item()}")
print(f"Window kept      : {keep_mask[-16:].all().item()}")
