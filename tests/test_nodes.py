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

from agents.nodes import AgentNodes, route_after_router


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
        "use_hyde": False,
        "router_reason": "",
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
# route_after_router (stateless, no dependencies)
# ---------------------------------------------------------------------------

def test_route_after_router_returns_hyde_node_when_true():
    assert route_after_router(make_state(use_hyde=True)) == "hyde_node"


def test_route_after_router_returns_retrieve_node_when_false():
    assert route_after_router(make_state(use_hyde=False)) == "retrieve_node"


# ---------------------------------------------------------------------------
# router_node
# ---------------------------------------------------------------------------

def test_router_node_parses_valid_json():
    nodes = make_nodes()
    nodes.llm = RunnableLambda(
        lambda _: AIMessage(content='{"use_hyde": true, "reason": "sentiment question"}')
    )
    result = nodes.router_node(make_state())

    assert result["use_hyde"] is True
    assert result["router_reason"] == "sentiment question"


def test_router_node_falls_back_on_invalid_json():
    """
    When the LLM returns something that isn't valid JSON, router_node should
    default to direct retrieval rather than crashing.
    """
    nodes = make_nodes()
    nodes.llm = RunnableLambda(lambda _: AIMessage(content="this is not json at all"))

    result = nodes.router_node(make_state())

    assert result["use_hyde"] is False
    assert "JSON parse failed" in result["router_reason"]


def test_router_node_falls_back_on_empty_response():
    nodes = make_nodes()
    nodes.llm = RunnableLambda(lambda _: AIMessage(content=""))

    result = nodes.router_node(make_state())

    assert result["use_hyde"] is False


def test_router_node_clears_rewritten_query():
    """router_node should always reset rewritten_query to empty string."""
    nodes = make_nodes()
    nodes.llm = RunnableLambda(
        lambda _: AIMessage(content='{"use_hyde": false, "reason": "factual"}')
    )
    result = nodes.router_node(make_state(rewritten_query="stale value from previous run"))

    assert result["rewritten_query"] == ""


# ---------------------------------------------------------------------------
# retrieve_node
# ---------------------------------------------------------------------------

def test_retrieve_node_uses_rewritten_query_when_hyde():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    nodes = make_nodes(retriever=mock_retriever)
    nodes.retrieve_node(make_state(use_hyde=True, rewritten_query="hypothetical post"))

    mock_retriever.retrieve.assert_called_once_with("hypothetical post", k=10)


def test_retrieve_node_uses_original_question_when_no_hyde():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    nodes = make_nodes(retriever=mock_retriever)
    nodes.retrieve_node(make_state(use_hyde=False, question="original question"))

    mock_retriever.retrieve.assert_called_once_with("original question", k=10)


def test_retrieve_node_falls_back_to_question_when_hyde_but_empty_rewrite():
    """
    If use_hyde=True but rewritten_query is empty (e.g. HyDE node failed),
    retrieve_node should fall back to the original question.
    """
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    nodes = make_nodes(retriever=mock_retriever)
    nodes.retrieve_node(make_state(use_hyde=True, rewritten_query="", question="fallback"))

    mock_retriever.retrieve.assert_called_once_with("fallback", k=10)


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
