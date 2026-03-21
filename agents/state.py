"""LangGraph shared state passed between all nodes."""

from typing import List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


class AgentState(TypedDict):
    question: str
    rewritten_query: str          # Set by hyde_node; empty string if skipped
    retrieved_docs: List[Document]
    answer: str
    use_hyde: bool                # Set by router_node (LLM decision)
    router_reason: str            # Router's explanation for the HyDE decision
