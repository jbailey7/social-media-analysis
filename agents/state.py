"""State dict passed between every node in the graph."""

from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document


class AgentState(TypedDict):
    question: str
    rewritten_query: str          # filled in by hyde_node; empty string until then
    retrieved_docs: List[Document]
    answer: str
