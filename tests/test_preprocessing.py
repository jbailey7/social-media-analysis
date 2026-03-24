"""
Tests for rag/preprocessing.py

clean_post() is a pure function with no external dependencies — no mocking needed.
"""

from rag.preprocessing import clean_post


# URL removal

def test_removes_http_url():
    assert clean_post("check this out http://example.com") == "check this out"


def test_removes_https_url():
    assert clean_post("go to https://example.com/page?q=1 now") == "go to now"


def test_removes_www_url():
    assert clean_post("visit www.example.com for more") == "visit for more"


def test_removes_url_with_path():
    assert clean_post("read https://news.site/article/123 today") == "read today"


def test_handles_only_url():
    assert clean_post("https://example.com") == ""


def test_removes_multiple_urls():
    result = clean_post("see http://a.com and https://b.com for details")
    assert "http" not in result
    assert "a.com" not in result
    assert "b.com" not in result


# Whitespace normalisation

def test_normalizes_multiple_spaces():
    assert clean_post("too   many    spaces") == "too many spaces"


def test_strips_leading_and_trailing_whitespace():
    assert clean_post("  hello world  ") == "hello world"


def test_normalizes_newlines():
    assert clean_post("line one\nline two") == "line one line two"


def test_normalizes_tabs():
    assert clean_post("word1\tword2") == "word1 word2"


# Edge cases

def test_handles_none():
    assert clean_post(None) == ""


def test_handles_empty_string():
    assert clean_post("") == ""


def test_non_string_input_is_cast_to_string():
    # clean_post casts input to str before processing
    assert clean_post(12345) == "12345"


# Preserved content

def test_preserves_hashtags():
    # clean_post does not remove hashtags — only URLs and excess whitespace
    result = clean_post("loving this #Bitcoin #crypto")
    assert "#Bitcoin" in result
    assert "#crypto" in result


def test_preserves_emojis():
    result = clean_post("great news 🚀🌙")
    assert "🚀" in result
    assert "🌙" in result


def test_preserves_normal_text():
    text = "People were very excited about the new product launch"
    assert clean_post(text) == text
