"""
Benchmark: KVShrinker compression ratio vs latency vs quality proxy.

Run:
    python benchmarks/benchmark_compression.py

Outputs a table comparing:
  - budget_ratio vs kept_tokens vs wall-clock time
  - H2O / SnapKV / RAG-aware scorer comparison
"""

import time
import torch

from vllm_kv_shrinker.core.kv_shrinker import KVShrinker
from vllm_kv_shrinker.core.config import KVShrinkerConfig, EvictionPolicy
from vllm_kv_shrinker.rag.rag_signal import RAGSignal


def make_batch(seq_len: int, num_heads: int = 32, num_kv_heads: int = 8, head_dim: int = 128):
    torch.manual_seed(0)
    key = torch.randn(seq_len, num_kv_heads, head_dim)
    val = torch.randn(seq_len, num_kv_heads, head_dim)
    raw = torch.rand(num_heads, seq_len, seq_len)
    attn = raw / raw.sum(dim=-1, keepdim=True)
    return key, val, attn


def benchmark_budget_ratios():
    print("\n=== Budget Ratio Sweep (RAG-Aware, seq_len=1024) ===")
    print(f"{'Budget':>8} {'Kept':>8} {'Ratio':>8} {'Time(ms)':>10}")
    print("-" * 40)
    key, val, attn = make_batch(seq_len=1024)
    for ratio in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        config = KVShrinkerConfig(budget_ratio=ratio, policy=EvictionPolicy.RAG_AWARE)
        shrinker = KVShrinker(config, layer_idx=16, total_layers=32)
        t0 = time.perf_counter()
        _, _, mask = shrinker.compress(key, val, attn)
        elapsed = (time.perf_counter() - t0) * 1000
        kept = mask.sum().item()
        print(f"{ratio:>8.0%} {kept:>8d} {kept/1024:>8.1%} {elapsed:>10.2f}")


def benchmark_policies():
    print("\n=== Policy Comparison (budget=30%, seq_len=2048) ===")
    print(f"{'Policy':>16} {'Kept':>8} {'Time(ms)':>10}")
    print("-" * 38)
    key, val, attn = make_batch(seq_len=2048)
    # RAG signal: 50 keyword tokens
    rag = RAGSignal.from_token_boosts({i: 1.0 for i in range(100, 150)})
    for policy in EvictionPolicy:
        config = KVShrinkerConfig(budget_ratio=0.3, policy=policy)
        shrinker = KVShrinker(config, layer_idx=16, total_layers=32)
        t0 = time.perf_counter()
        _, _, mask = shrinker.compress(
            key, val, attn,
            rag_signal=rag if policy == EvictionPolicy.RAG_AWARE else None,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        kept = mask.sum().item()
        print(f"{policy.value:>16} {kept:>8d} {elapsed:>10.2f}")


def benchmark_seq_lengths():
    print("\n=== Sequence Length Scaling (RAG-Aware, budget=30%) ===")
    print(f"{'SeqLen':>8} {'Kept':>8} {'Time(ms)':>10} {'Mem(MB)':>10}")
    print("-" * 42)
    for seq_len in [256, 512, 1024, 2048, 4096, 8192]:
        key, val, attn = make_batch(seq_len=seq_len)
        config = KVShrinkerConfig(budget_ratio=0.3, policy=EvictionPolicy.RAG_AWARE)
        shrinker = KVShrinker(config, layer_idx=0, total_layers=32)
        t0 = time.perf_counter()
        pk, pv, mask = shrinker.compress(key, val, attn)
        elapsed = (time.perf_counter() - t0) * 1000
        kept = mask.sum().item()
        # KV memory: 2 tensors × kept × num_kv_heads × head_dim × 4 bytes
        mem_mb = 2 * kept * 8 * 128 * 4 / 1e6
        print(f"{seq_len:>8d} {kept:>8d} {elapsed:>10.2f} {mem_mb:>10.2f}")


def benchmark_rag_retention():
    print("\n=== RAG Keyword Retention Rate (seq_len=512, 30 keywords) ===")
    print(f"{'Policy':>16} {'KW Retained':>12} {'Retention%':>12}")
    print("-" * 44)
    seq_len = 512
    kw_indices = list(range(100, 130))   # 30 keywords at positions 100-129
    rag = RAGSignal.from_token_boosts({i: 1.0 for i in kw_indices})
    key, val, attn = make_batch(seq_len=seq_len)

    for policy in EvictionPolicy:
        config = KVShrinkerConfig(budget_ratio=0.3, policy=policy)
        shrinker = KVShrinker(config, layer_idx=0, total_layers=32)
        _, _, mask = shrinker.compress(
            key, val, attn,
            rag_signal=rag if policy == EvictionPolicy.RAG_AWARE else None,
        )
        retained = sum(mask[i].item() for i in kw_indices)
        pct = retained / len(kw_indices) * 100
        print(f"{policy.value:>16} {retained:>12d} {pct:>11.1f}%")


if __name__ == "__main__":
    benchmark_budget_ratios()
    benchmark_policies()
    benchmark_seq_lengths()
    benchmark_rag_retention()
