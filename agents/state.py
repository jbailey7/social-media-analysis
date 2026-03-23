"""LangGraph shared state passed between all nodes."""

from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document


class AgentState(TypedDict):
    question: str
    rewritten_query: str          # Set by hyde_node; empty string before HyDE runs
    retrieved_docs: List[Document]
    answer: str
