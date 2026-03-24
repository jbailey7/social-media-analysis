"""
Tests for safe_metadata() in scripts/ingest.py

safe_metadata() is a pure function that sanitises post metadata before
upserting to Pinecone. No API calls are made in any of these tests.
"""

import pytest
from scripts.ingest import safe_metadata


# None and missing values

def test_excludes_none_values():
    result = safe_metadata({"key": None, "other": "value"})
    assert "key" not in result
    assert result["other"] == "value"


def test_handles_none_input():
    assert safe_metadata(None) == {}


def test_handles_empty_dict():
    assert safe_metadata({}) == {}


# Accepted types

def test_keeps_string_values():
    assert safe_metadata({"theme": "Cryptocurrency"})["theme"] == "Cryptocurrency"


def test_keeps_int_values():
    assert safe_metadata({"count": 42})["count"] == 42


def test_keeps_float_values():
    assert safe_metadata({"score": 0.95})["score"] == 0.95


def test_keeps_bool_true():
    assert safe_metadata({"active": True})["active"] is True


def test_keeps_bool_false():
    assert safe_metadata({"active": False})["active"] is False


def test_keeps_zero_int():
    assert safe_metadata({"n": 0})["n"] == 0


def test_keeps_empty_string():
    assert safe_metadata({"lang": ""})["lang"] == ""


# Non-serializable types are stringified

def test_stringifies_list():
    result = safe_metadata({"tags": ["a", "b", "c"]})
    assert result["tags"] == "['a', 'b', 'c']"


def test_stringifies_dict():
    result = safe_metadata({"nested": {"x": 1}})
    assert isinstance(result["nested"], str)


def test_stringifies_tuple():
    result = safe_metadata({"pair": (1, 2)})
    assert isinstance(result["pair"], str)


# Mixed metadata dict

def test_handles_mixed_types():
    result = safe_metadata({
        "theme":     "AI",
        "score":     0.9,
        "tags":      ["ml", "llm"],
        "timestamp": None,
    })
    assert result["theme"] == "AI"
    assert result["score"] == 0.9
    assert result["tags"] == "['ml', 'llm']"
    assert "timestamp" not in result


def test_all_none_values_returns_empty():
    result = safe_metadata({"a": None, "b": None, "c": None})
    assert result == {}
