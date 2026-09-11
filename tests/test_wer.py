"""
tests/test_wer.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for WER calculation.
"""

import pytest
from evaluation.wer import calculate_wer, normalize_text_for_wer


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text_for_wer("Hello World") == "hello world"

    def test_removes_punctuation(self):
        result = normalize_text_for_wer("Hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_collapses_whitespace(self):
        result = normalize_text_for_wer("hello   world")
        assert result == "hello world"

    def test_strips(self):
        result = normalize_text_for_wer("  hello  ")
        assert result == "hello"


class TestCalculateWER:
    def test_perfect_match(self):
        result = calculate_wer("hello world", "hello world")
        assert result.wer == 0.0
        assert result.substitutions == 0
        assert result.deletions == 0
        assert result.insertions == 0

    def test_one_substitution(self):
        result = calculate_wer("hello world", "hello word")
        assert result.substitutions == 1
        assert result.wer == pytest.approx(0.5)

    def test_one_deletion(self):
        result = calculate_wer("hello world today", "hello today")
        assert result.deletions == 1

    def test_one_insertion(self):
        result = calculate_wer("hello world", "hello uh world")
        assert result.insertions == 1

    def test_empty_reference_raises(self):
        with pytest.raises(ValueError):
            calculate_wer("", "hello world")

    def test_word_count(self):
        result = calculate_wer("one two three four five", "one two three four five")
        assert result.reference_word_count == 5

    def test_case_insensitive(self):
        # WER normalizes before comparison
        result = calculate_wer("Hello World", "hello world")
        assert result.wer == 0.0
