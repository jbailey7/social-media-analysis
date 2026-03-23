"""
LangGraph node functions.

Nodes:
  router_node  — gpt-4o-mini decides whether to use HyDE
  hyde_node    — gpt-4o-mini generates a hypothetical social media post
  retrieve_node — Pinecone retrieval using rewritten or original query
  answer_node  — fine-tuned SmolLM2-360M generates the final answer
"""

import json
import os

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from agents.state import AgentState
from agents.prompts import ROUTER_PROMPT, HYDE_PROMPT
from models.smollm import generate_summary

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# --- Node 1: Router ---

def router_node(state: AgentState) -> AgentState:
    """
    Uses gpt-4o-mini to decide whether HyDE rewriting should be applied.
    Returns updated state with use_hyde and router_reason set.
    """
    chain = ROUTER_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"question": state["question"]})

    try:
        parsed = json.loads(raw)
        use_hyde = bool(parsed.get("use_hyde", False))
        reason = parsed.get("reason", "")
    except (json.JSONDecodeError, KeyError):
        # If the LLM fails to return valid JSON, default to direct retrieval
        use_hyde = False
        reason = "JSON parse failed; defaulting to direct retrieval"

    return {
        **state,
        "use_hyde": use_hyde,
        "router_reason": reason,
        "rewritten_query": "",
    }


# --- Node 2: HyDE ---

def hyde_node(state: AgentState) -> AgentState:
    """
    Uses gpt-4o-mini to generate a hypothetical social media post.
    Only called when router_node sets use_hyde=True.
    """
    chain = HYDE_PROMPT | llm | StrOutputParser()
    hypothetical = chain.invoke({"question": state["question"]})
    return {**state, "rewritten_query": hypothetical}


# --- Node 3: Retriever ---

_retriever = None

def _get_retriever():
    """Lazy-load the PineconeRetriever so it's only built once."""
    global _retriever
    if _retriever is None:
        from rag.retriever import PineconeRetriever
        _retriever = PineconeRetriever()
    return _retriever


def retrieve_node(state: AgentState) -> AgentState:
    """
    Retrieves top-10 relevant social media posts from Pinecone.
    Uses the rewritten query if HyDE was applied, otherwise the original question.
    """
    query = state["rewritten_query"] if state.get("use_hyde") and state["rewritten_query"] else state["question"]
    retriever = _get_retriever()
    docs = retriever.retrieve(query, k=10)
    return {**state, "retrieved_docs": docs}


# --- Node 4: Answer (fine-tuned SmolLM2-360M) ---

_ft_model = None
_ft_tokenizer = None

def _get_ft_model():
    """Lazy-load the fine-tuned SmolLM2 so it's only built once."""
    global _ft_model, _ft_tokenizer
    if _ft_model is None:
        from models.smollm import load_model
        _ft_model, _ft_tokenizer = load_model()
    return _ft_model, _ft_tokenizer


def answer_node(state: AgentState) -> AgentState:
    """
    Formats retrieved posts and passes them to fine-tuned SmolLM2 to generate
    the final answer.

    The format matches the training data produced by generate_training_data.py:
      'Here are social media posts from December 2024:\\n\\nPost 1: ...\\n\\nQuestion: ...'
    """
    docs = state["retrieved_docs"]
    posts_text = "\n".join(
        f"Post {i+1}: {doc.page_content}" for i, doc in enumerate(docs[:10])
    )
    context = (
        "Here are social media posts from December 2024:\n\n"
        f"{posts_text}\n\n"
        f"Question: {state['question']}"
    )

    model, tokenizer = _get_ft_model()
    answer = generate_summary(model, tokenizer, context, max_new_tokens=150)
    return {**state, "answer": answer}


# --- Conditional edge function ---

def route_after_router(state: AgentState) -> str:
    """Returns the next node name based on the router's decision."""
    return "hyde_node" if state.get("use_hyde") else "retrieve_node"
