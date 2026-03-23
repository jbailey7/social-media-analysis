"""
Tests for agents/nodes.py

AgentNodes takes all dependencies via its constructor, making it straightforward
to test with lightweight mocks. No API calls are made in any of these tests.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from agents.nodes import AgentNodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(**kwargs) -> dict:
    """Return a minimal AgentState dict with sensible defaults."""
    base = {
        "question": "test question",
        "rewritten_query": "",
        "retrieved_docs": [],
        "answer": "",
    }
    return {**base, **kwargs}


def make_nodes(retriever=None, model=None, tokenizer=None) -> AgentNodes:
    """Return an AgentNodes instance with mock dependencies."""
    return AgentNodes(
        retriever=retriever or MagicMock(),
        model=model    or MagicMock(),
        tokenizer=tokenizer or MagicMock(),
    )


# ---------------------------------------------------------------------------
# hyde_node
# ---------------------------------------------------------------------------

def test_hyde_node_sets_rewritten_query():
    nodes = make_nodes()
    nodes.llm = RunnableLambda(lambda _: AIMessage(content="just bought more $BTC lol"))

    result = nodes.hyde_node(make_state(question="What did people say about Bitcoin?"))

    assert result["rewritten_query"] == "just bought more $BTC lol"


def test_hyde_node_preserves_other_state_fields():
    nodes = make_nodes()
    nodes.llm = RunnableLambda(lambda _: AIMessage(content="hypothetical post"))

    result = nodes.hyde_node(make_state(question="test?", answer="existing"))

    assert result["question"] == "test?"
    assert result["answer"] == "existing"


# ---------------------------------------------------------------------------
# retrieve_node
# ---------------------------------------------------------------------------

def test_retrieve_node_uses_rewritten_query():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    nodes = make_nodes(retriever=mock_retriever)
    nodes.retrieve_node(make_state(rewritten_query="hypothetical post"))

    mock_retriever.retrieve.assert_called_once_with("hypothetical post", k=10)


def test_retrieve_node_falls_back_to_question_when_rewrite_empty():
    """
    If rewritten_query is empty (e.g. hyde_node failed), retrieve_node
    should fall back to the original question rather than querying with
    an empty string.
    """
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    nodes = make_nodes(retriever=mock_retriever)
    nodes.retrieve_node(make_state(rewritten_query="", question="original question"))

    mock_retriever.retrieve.assert_called_once_with("original question", k=10)


def test_retrieve_node_returns_docs_in_state():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [Document(page_content="a post", metadata={})]

    nodes = make_nodes(retriever=mock_retriever)
    result = nodes.retrieve_node(make_state(rewritten_query="query"))

    assert len(result["retrieved_docs"]) == 1
    assert result["retrieved_docs"][0].page_content == "a post"


# ---------------------------------------------------------------------------
# answer_node
# ---------------------------------------------------------------------------

def test_answer_node_formats_context_correctly():
    """
    answer_node must format posts and question in a way that matches the
    training data produced by generate_training_data.py.
    """
    nodes = make_nodes()
    docs = [
        Document(page_content="first post", metadata={}),
        Document(page_content="second post", metadata={}),
    ]
    state = make_state(question="what happened?", retrieved_docs=docs)

    with patch("agents.nodes.generate_summary", return_value="test answer") as mock_gen:
        result = nodes.answer_node(state)

        context = mock_gen.call_args[0][2]
        assert "Here are social media posts from December 2024:" in context
        assert "Post 1: first post" in context
        assert "Post 2: second post" in context
        assert "Question: what happened?" in context
        assert result["answer"] == "test answer"


def test_answer_node_caps_docs_at_ten():
    """answer_node should use at most 10 posts regardless of how many are retrieved."""
    nodes = make_nodes()
    docs = [Document(page_content=f"post {i}", metadata={}) for i in range(15)]
    state = make_state(question="test?", retrieved_docs=docs)

    with patch("agents.nodes.generate_summary", return_value="answer") as mock_gen:
        nodes.answer_node(state)

        context = mock_gen.call_args[0][2]
        assert "Post 10:" in context
        assert "Post 11:" not in context
