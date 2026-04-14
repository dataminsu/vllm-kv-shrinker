"""
Eviction policies: given importance scores, select which token indices to keep.

Each policy returns a boolean mask [seq_len] where True = keep the token.
Sink tokens and recent-window tokens are always kept regardless of scores.
"""

from __future__ import annotations

import torch


def select_tokens_to_keep(
    scores: torch.Tensor,
    budget: int,
    sink_size: int = 4,
    window_size: int = 32,
    min_budget: int = 16,
) -> torch.Tensor:
    """
    Core selection logic shared by all policies.

    Always preserves:
        1. First `sink_size` tokens (attention sinks).
        2. Last `window_size` tokens (recency bias).
        3. Top-(budget - sink_size - window_size) tokens by score from the rest.

    Args:
        scores:      [seq_len] importance scores (higher = keep).
        budget:      Total number of tokens to keep.
        sink_size:   Number of initial tokens always kept.
        window_size: Number of trailing tokens always kept.
        min_budget:  Hard lower bound on budget.

    Returns:
        keep_mask: [seq_len] boolean tensor, True = keep.
    """
    seq_len = scores.shape[0]
    budget = max(budget, min_budget)
    budget = min(budget, seq_len)

    keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=scores.device)

    # --- 1. Sink tokens ---
    actual_sink = min(sink_size, seq_len)
    keep_mask[:actual_sink] = True

    # --- 2. Recent window ---
    actual_window = min(window_size, seq_len)
    keep_mask[-actual_window:] = True

    already_kept = keep_mask.sum().item()
    remaining_budget = budget - already_kept

    if remaining_budget <= 0:
        return keep_mask

    # --- 3. Top-k by score from the middle region ---
    middle_mask = ~keep_mask
    middle_indices = middle_mask.nonzero(as_tuple=True)[0]

    if middle_indices.numel() == 0:
        return keep_mask

    middle_scores = scores[middle_indices]
    k = min(remaining_budget, middle_indices.numel())
    _, top_local = middle_scores.topk(k)
    top_global = middle_indices[top_local]
    keep_mask[top_global] = True

    return keep_mask


def apply_eviction(
    scores: torch.Tensor,
    budget_ratio: float,
    sink_size: int = 4,
    window_size: int = 32,
    min_budget: int = 16,
) -> torch.Tensor:
    """
    Convenience wrapper: convert budget_ratio → integer budget → keep_mask.

    Args:
        scores:       [seq_len] importance scores.
        budget_ratio: Fraction of tokens to keep (0 < r <= 1.0).
        sink_size:    Always-kept initial tokens.
        window_size:  Always-kept trailing tokens.
        min_budget:   Minimum number of tokens to keep.

    Returns:
        keep_mask: [seq_len] boolean tensor.
    """
    seq_len = scores.shape[0]
    budget = max(int(seq_len * budget_ratio), min_budget)
    return select_tokens_to_keep(
        scores,
        budget=budget,
        sink_size=sink_size,
        window_size=window_size,
        min_budget=min_budget,
    )


def apply_eviction_to_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    keep_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Filter key/value tensors using a boolean keep mask.

    Args:
        key:       [seq_len, num_heads, head_dim]
        value:     [seq_len, num_heads, head_dim]
        keep_mask: [seq_len] boolean mask

    Returns:
        (pruned_key, pruned_value, kept_indices)
        pruned_key:    [kept, num_heads, head_dim]
        pruned_value:  [kept, num_heads, head_dim]
        kept_indices:  [kept] original token positions
    """
    kept_indices = keep_mask.nonzero(as_tuple=True)[0]
    return key[kept_indices], value[kept_indices], kept_indices
