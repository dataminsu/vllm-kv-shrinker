"""
Tokenization utilities for mapping RAG keywords to token positions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


def keywords_to_token_boosts(
    keywords: Dict[str, float],
    tokenizer,
    input_ids: torch.Tensor,
    dense_scores: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a keyword → score dict to per-token boost tensors.

    Args:
        keywords:     {keyword_string: lexical_score}  (scores in [0,1])
        tokenizer:    HuggingFace tokenizer.
        input_ids:    [seq_len] token id tensor.
        dense_scores: Optional {keyword: dense_score}.

    Returns:
        lexical_boost: [seq_len] float tensor
        dense_boost:   [seq_len] float tensor
    """
    seq_len = input_ids.shape[0]
    lexical = torch.zeros(seq_len, dtype=torch.float32)
    dense = torch.zeros(seq_len, dtype=torch.float32)

    dense_scores = dense_scores or {}
    id_list: List[int] = input_ids.tolist()

    for keyword, lex_score in keywords.items():
        kw_ids = tokenizer.encode(keyword, add_special_tokens=False)
        if not kw_ids:
            continue
        positions = _find_subseq(id_list, kw_ids)
        den_score = dense_scores.get(keyword, 0.0)
        for start in positions:
            for offset in range(len(kw_ids)):
                idx = start + offset
                lexical[idx] = max(lexical[idx].item(), lex_score)
                dense[idx] = max(dense[idx].item(), den_score)

    return lexical, dense


def normalize_bm25_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize BM25 scores to [0, 1] via min-max scaling."""
    if not scores:
        return scores
    min_s, max_s = min(scores.values()), max(scores.values())
    if max_s == min_s:
        return {k: 1.0 for k in scores}
    return {k: (v - min_s) / (max_s - min_s) for k, v in scores.items()}


def _find_subseq(sequence: List[int], subseq: List[int]) -> List[int]:
    """Return start indices of all occurrences of subseq in sequence."""
    n, m = len(sequence), len(subseq)
    if m == 0:
        return []
    return [i for i in range(n - m + 1) if sequence[i: i + m] == subseq]
