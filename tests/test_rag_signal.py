"""Tests for RAGSignal construction and token boost extraction."""

import pytest
import torch

from vllm_kv_shrinker.rag.rag_signal import RAGKeywordScore, RAGSignal, _find_subseq


class TestRAGKeywordScore:
    def test_combined_score(self):
        kw = RAGKeywordScore("Paris", lexical_score=0.8, dense_score=0.6)
        assert kw.combined_score(alpha=1.0, beta=0.5) == pytest.approx(0.8 + 0.3)


class TestRAGSignalFromTokenBoosts:
    def test_basic(self):
        sig = RAGSignal.from_token_boosts({5: 1.0, 10: 0.7})
        lex, dense = sig.get_token_boosts(20, torch.device("cpu"))
        assert lex[5].item() == pytest.approx(1.0)
        assert lex[10].item() == pytest.approx(0.7)
        assert lex[0].item() == pytest.approx(0.0)

    def test_dense_none_when_all_zero(self):
        sig = RAGSignal.from_token_boosts({3: 0.5})
        lex, dense = sig.get_token_boosts(10, torch.device("cpu"))
        assert lex is not None
        assert dense is None  # dense all-zero → returns None

    def test_empty_boosts(self):
        sig = RAGSignal()
        lex, dense = sig.get_token_boosts(32, torch.device("cpu"))
        assert lex is None
        assert dense is None

    def test_out_of_range_index_ignored(self):
        sig = RAGSignal.from_token_boosts({100: 1.0})
        lex, _ = sig.get_token_boosts(32, torch.device("cpu"))  # seq_len=32
        # Token 100 is out of range → all zeros
        assert lex.sum().item() == pytest.approx(0.0)


class TestRAGSignalFromKeywords:
    def test_keyword_matching(self):
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                # Simple char-level fake encoding
                return [ord(c) for c in text.lower() if c.isalpha()]

        # input_ids contains ord values for "ab" at two positions
        # Sequence: ..., 97, 98, ..., 97, 98, ...
        ids = torch.tensor([0, ord('a'), ord('b'), 5, ord('a'), ord('b'), 9])

        sig = RAGSignal.from_keywords(
            {"ab": 0.9},
            tokenizer=FakeTokenizer(),
            input_ids=ids,
        )
        lex, _ = sig.get_token_boosts(len(ids), torch.device("cpu"))
        # positions 1,2 and 4,5 should be boosted
        assert lex[1].item() == pytest.approx(0.9)
        assert lex[2].item() == pytest.approx(0.9)
        assert lex[4].item() == pytest.approx(0.9)
        assert lex[5].item() == pytest.approx(0.9)
        assert lex[0].item() == pytest.approx(0.0)

    def test_no_match(self):
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [999, 888]  # never appears in input

        ids = torch.tensor([1, 2, 3, 4, 5])
        sig = RAGSignal.from_keywords({"xyz": 1.0}, FakeTokenizer(), ids)
        lex, _ = sig.get_token_boosts(5, torch.device("cpu"))
        assert lex is not None
        assert lex.sum().item() == pytest.approx(0.0)


class TestFindSubseq:
    def test_single_match(self):
        assert _find_subseq([1, 2, 3, 4, 5], [3, 4]) == [2]

    def test_multiple_matches(self):
        assert _find_subseq([1, 2, 1, 2, 1], [1, 2]) == [0, 2]

    def test_no_match(self):
        assert _find_subseq([1, 2, 3], [4, 5]) == []

    def test_empty_subseq(self):
        assert _find_subseq([1, 2, 3], []) == []

    def test_subseq_longer_than_seq(self):
        assert _find_subseq([1, 2], [1, 2, 3]) == []
