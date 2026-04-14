"""
RAG Signal: bridges the retrieval pipeline with the KV shrinker.

The RAGSignal class carries per-token importance boosts derived from
retrieval keyword scores.  It is created by the RAG layer and passed
to vLLM's SamplingParams (or a forward context), where KVShrinker
reads it during attention computation.

Workflow:
    1. RAG retriever runs BM25/DPR, returns keyword → score dict.
    2. RAGSignal.from_keywords() tokenizes keywords and records positions.
    3. KVShrinker.compress() calls rag_signal.get_token_boosts(seq_len)
       to get a [seq_len] importance boost tensor.
    4. RAGAwareScorer adds the boost to the attention-based score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class RAGKeywordScore:
    """A single keyword and its retrieval importance score."""

    keyword: str
    lexical_score: float = 1.0   # BM25 / TF-IDF score (normalised to [0,1])
    dense_score: float = 0.0     # Dense retriever cosine similarity [0,1]

    def combined_score(self, alpha: float = 1.0, beta: float = 0.5) -> float:
        return alpha * self.lexical_score + beta * self.dense_score


class RAGSignal:
    """
    Carries retrieval-derived token importance boosts for one inference request.

    After tokenization, each keyword is mapped to one or more token indices.
    get_token_boosts() returns a dense [seq_len] tensor that the scorer uses.
    """

    def __init__(self) -> None:
        # mapping: token_index -> (lexical_boost, dense_boost)
        self._token_boosts: Dict[int, Tuple[float, float]] = {}
        self._seq_len: Optional[int] = None
        # True once from_keywords() is called — distinguishes "no boosts found"
        # from "no keywords provided at all" so get_token_boosts can return
        # zero tensors (signal present, nothing matched) vs None (no signal).
        self._keywords_provided: bool = False

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_keywords(
        cls,
        keywords: Dict[str, float],
        tokenizer,
        input_ids: Optional[torch.Tensor] = None,
        dense_scores: Optional[Dict[str, float]] = None,
    ) -> "RAGSignal":
        """
        Build a RAGSignal by tokenizing keywords and finding their positions
        in the input token sequence.

        Args:
            keywords:     {keyword: lexical_score}.  Scores should be in [0,1].
            tokenizer:    HuggingFace tokenizer (or any with encode() method).
            input_ids:    [seq_len] tensor of token ids for the full prompt.
                          If None, boosts are recorded by keyword token ids only.
            dense_scores: Optional {keyword: dense_retrieval_score}.

        Returns:
            RAGSignal with per-token boosts populated.
        """
        sig = cls()
        sig._keywords_provided = bool(keywords)
        dense_scores = dense_scores or {}

        for keyword, lex_score in keywords.items():
            dense = dense_scores.get(keyword, 0.0)
            # Tokenize the keyword (without special tokens)
            kw_ids = tokenizer.encode(keyword, add_special_tokens=False)

            if input_ids is not None:
                # Find all occurrences of kw_ids as a sub-sequence in input_ids
                positions = _find_subseq(input_ids, kw_ids)
                for pos in positions:
                    for offset in range(len(kw_ids)):
                        idx = pos + offset
                        prev_lex, prev_dense = sig._token_boosts.get(idx, (0.0, 0.0))
                        sig._token_boosts[idx] = (
                            max(prev_lex, lex_score),
                            max(prev_dense, dense),
                        )
            else:
                # No input_ids: record by token id (applied during get_token_boosts
                # if input_ids are provided later via set_input_ids)
                for tid in kw_ids:
                    sig._token_boosts[tid] = (
                        max(sig._token_boosts.get(tid, (0.0, 0.0))[0], lex_score),
                        max(sig._token_boosts.get(tid, (0.0, 0.0))[1], dense),
                    )

        return sig

    @classmethod
    def from_token_boosts(
        cls,
        token_boosts: Dict[int, float],
        dense_boosts: Optional[Dict[int, float]] = None,
    ) -> "RAGSignal":
        """
        Build directly from pre-computed per-token boost dicts.

        Args:
            token_boosts:  {token_index: lexical_boost}
            dense_boosts:  {token_index: dense_boost}  (optional)
        """
        sig = cls()
        dense_boosts = dense_boosts or {}
        for idx, lex in token_boosts.items():
            sig._token_boosts[idx] = (lex, dense_boosts.get(idx, 0.0))
        return sig

    def set_input_ids(self, input_ids: torch.Tensor) -> None:
        """
        Remap token-id-based boosts to position-based boosts once we know
        the full input sequence.  Only needed when from_keywords was called
        without input_ids.
        """
        self._seq_len = input_ids.shape[0]
        new_boosts: Dict[int, Tuple[float, float]] = {}
        id_list = input_ids.tolist()
        for pos, tid in enumerate(id_list):
            if tid in self._token_boosts:
                prev = new_boosts.get(pos, (0.0, 0.0))
                cur = self._token_boosts[tid]
                new_boosts[pos] = (max(prev[0], cur[0]), max(prev[1], cur[1]))
        self._token_boosts = new_boosts

    # ------------------------------------------------------------------
    # Main accessor
    # ------------------------------------------------------------------

    def get_token_boosts(
        self,
        seq_len: int,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Return (lexical_boost, dense_boost) tensors of shape [seq_len].

        Returns (None, None) if no boosts are recorded.
        """
        if not self._token_boosts:
            if self._keywords_provided:
                # Keywords were given but none matched — return explicit zeros
                # so downstream knows a RAG signal was active (just no matches).
                z = torch.zeros(seq_len, dtype=torch.float32, device=device)
                return z, None
            return None, None

        lexical = torch.zeros(seq_len, dtype=torch.float32, device=device)
        dense = torch.zeros(seq_len, dtype=torch.float32, device=device)

        for idx, (lex, den) in self._token_boosts.items():
            if idx < seq_len:
                lexical[idx] = lex
                dense[idx] = den

        has_dense = dense.any().item()
        return lexical, (dense if has_dense else None)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def num_boosted_tokens(self) -> int:
        return len(self._token_boosts)

    def boosted_indices(self) -> List[int]:
        return sorted(self._token_boosts.keys())

    def __repr__(self) -> str:
        return f"RAGSignal(boosted_tokens={self.num_boosted_tokens()})"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _find_subseq(
    sequence: torch.Tensor, subseq: List[int]
) -> List[int]:
    """
    Find all starting positions of `subseq` in `sequence` (1-D int tensor).
    Returns a list of starting indices.
    """
    if not subseq:
        return []
    seq_list = sequence.tolist() if isinstance(sequence, torch.Tensor) else list(sequence)
    n, m = len(seq_list), len(subseq)
    positions = []
    for i in range(n - m + 1):
        if seq_list[i : i + m] == subseq:
            positions.append(i)
    return positions
