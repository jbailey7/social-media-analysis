"""
LangGraph node functions.

All node methods live on AgentNodes, which takes its dependencies (retriever,
model, tokenizer) via constructor injection. This avoids module-level mutable
globals and makes the nodes straightforward to test with mock dependencies.

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
from rag.retriever import PineconeRetriever


class AgentNodes:
    """
    Holds injected dependencies and exposes LangGraph-compatible node methods.

    Args:
        retriever:  Initialised PineconeRetriever for post lookup.
        model:      Loaded SmolLM2-360M (base or LoRA fine-tuned).
        tokenizer:  Tokenizer matching the model.
    """

    def __init__(self, retriever: PineconeRetriever, model, tokenizer):
        self.retriever = retriever
        self.model = model
        self.tokenizer = tokenizer
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    # --- Node 1: Router ---

    def router_node(self, state: AgentState) -> AgentState:
        """
        Uses gpt-4o-mini to decide whether HyDE rewriting should be applied.
        Returns updated state with use_hyde and router_reason set.
        """
        chain = ROUTER_PROMPT | self.llm | StrOutputParser()
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

    def hyde_node(self, state: AgentState) -> AgentState:
        """
        Uses gpt-4o-mini to generate a hypothetical social media post.
        Only called when router_node sets use_hyde=True.
        """
        chain = HYDE_PROMPT | self.llm | StrOutputParser()
        hypothetical = chain.invoke({"question": state["question"]})
        return {**state, "rewritten_query": hypothetical}

    # --- Node 3: Retriever ---

    def retrieve_node(self, state: AgentState) -> AgentState:
        """
        Retrieves top-10 relevant social media posts from Pinecone.
        Uses the rewritten query if HyDE was applied, otherwise the original question.
        """
        query = (
            state["rewritten_query"]
            if state.get("use_hyde") and state["rewritten_query"]
            else state["question"]
        )
        docs = self.retriever.retrieve(query, k=10)
        return {**state, "retrieved_docs": docs}

    # --- Node 4: Answer (fine-tuned SmolLM2-360M) ---

    def answer_node(self, state: AgentState) -> AgentState:
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
        answer = generate_summary(self.model, self.tokenizer, context, max_new_tokens=150)
        return {**state, "answer": answer}


# --- Conditional edge function (stateless, no dependencies needed) ---

def route_after_router(state: AgentState) -> str:
    """Returns the next node name based on the router's decision."""
    return "hyde_node" if state.get("use_hyde") else "retrieve_node"
