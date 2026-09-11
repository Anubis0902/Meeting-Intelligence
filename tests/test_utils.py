"""
tests/test_utils.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for src/utils.py

Run with:
    pytest tests/ -v
"""

import pytest
from src.utils import seconds_to_hms, hms_to_seconds, chunk_text_by_words, count_words


class TestSecondsToHms:
    def test_zero(self):
        assert seconds_to_hms(0.0) == "00:00:00"

    def test_seconds_only(self):
        assert seconds_to_hms(45.0) == "00:00:45"

    def test_minutes(self):
        assert seconds_to_hms(90.0) == "00:01:30"

    def test_hours(self):
        assert seconds_to_hms(3661.0) == "01:01:01"

    def test_float_truncated(self):
        # Float seconds are truncated (not rounded)
        assert seconds_to_hms(59.9) == "00:00:59"


class TestHmsToSeconds:
    def test_zero(self):
        assert hms_to_seconds("00:00:00") == 0.0

    def test_minutes(self):
        assert hms_to_seconds("00:01:30") == 90.0

    def test_hours(self):
        assert hms_to_seconds("01:01:01") == 3661.0

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            hms_to_seconds("01:30")  # missing hours component


class TestChunkText:
    def test_short_text_no_chunking(self):
        text = "hello world"
        chunks = chunk_text_by_words(text, max_words=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_exact_split(self):
        text = "a b c d e f"  # 6 words
        chunks = chunk_text_by_words(text, max_words=3)
        assert len(chunks) == 2
        assert chunks[0] == "a b c"
        assert chunks[1] == "d e f"

    def test_uneven_split(self):
        text = "a b c d e"  # 5 words
        chunks = chunk_text_by_words(text, max_words=3)
        assert len(chunks) == 2
        assert chunks[1] == "d e"


class TestCountWords:
    def test_basic(self):
        assert count_words("hello world") == 2

    def test_empty(self):
        assert count_words("") == 0  # "".split() returns [] → 0 words

    def test_multi_space(self):
        # count_words splits on whitespace — multiple spaces count as one
        assert count_words("hello  world") == 2
