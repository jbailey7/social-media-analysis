"""
LangGraph node functions.

All node methods live on AgentNodes, which takes its dependencies (retriever,
model, tokenizer) via constructor injection. This avoids module-level mutable
globals and makes the nodes straightforward to test with mock dependencies.

Nodes:
  hyde_node     — gpt-4o-mini generates a hypothetical social media post
  retrieve_node — Pinecone retrieval using the HyDE-rewritten query
  answer_node   — fine-tuned SmolLM2-360M generates the final answer

HyDE is always applied. Experiments showed that always generating a
hypothetical post before retrieval consistently outperforms conditional
routing (see notebooks/rag_experiments.ipynb, Experiment 2).
"""

import os

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from agents.state import AgentState
from agents.prompts import HYDE_PROMPT
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

    # --- Node 1: HyDE ---

    def hyde_node(self, state: AgentState) -> AgentState:
        """
        Uses gpt-4o-mini to generate a hypothetical social media post that
        resembles what a relevant result would look like. Embedding this post
        instead of the raw question improves retrieval because the index
        contains posts, not questions.
        """
        chain = HYDE_PROMPT | self.llm | StrOutputParser()
        hypothetical = chain.invoke({"question": state["question"]})
        return {**state, "rewritten_query": hypothetical}

    # --- Node 2: Retriever ---

    def retrieve_node(self, state: AgentState) -> AgentState:
        """
        Retrieves top-10 relevant social media posts from Pinecone.
        Uses the HyDE-rewritten query; falls back to the original question
        if rewritten_query is empty.
        """
        query = state["rewritten_query"] if state["rewritten_query"] else state["question"]
        docs = self.retriever.retrieve(query, k=10)
        return {**state, "retrieved_docs": docs}

    # --- Node 3: Answer (fine-tuned SmolLM2-360M) ---

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
