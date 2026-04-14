# vllm-kv-shrinker

RAG-aware KV cache pruning for vLLM. Protects semantically important tokens
using retrieval signals from BM25/dense retrievers.

## Install

```bash
pip install -e .
```

## Quick Start

```python
from vllm_kv_shrinker import KVShrinker, KVShrinkerConfig, RAGSignal

config = KVShrinkerConfig(budget_ratio=0.3, policy="rag_aware")
shrinker = KVShrinker(config, layer_idx=16, total_layers=32)

rag = RAGSignal.from_token_boosts({40: 1.0, 41: 1.0, 80: 0.9})
pruned_key, pruned_val, mask = shrinker.compress(key, value, attn_weights, rag_signal=rag)
```

## Tests

```bash
pytest tests/ -v
```
