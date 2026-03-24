"""
The three nodes that make up the pipeline.

Dependencies are injected via the AgentNodes constructor rather than stored
as module-level globals, which makes them easy to swap out in tests.
"""

import os

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from agents.state import AgentState
from agents.prompts import HYDE_PROMPT
from models.smollm import generate_summary
from rag.retriever import PineconeRetriever
from config import CHAT_MODEL


class AgentNodes:
    """
    Holds the retriever, model, and tokenizer and exposes them as graph nodes.

    Pass in a base or fine-tuned model — the nodes don't care which.
    """

    def __init__(self, retriever: PineconeRetriever, model, tokenizer):
        self.retriever = retriever
        self.model = model
        self.tokenizer = tokenizer
        self.llm = ChatOpenAI(
            model=CHAT_MODEL,
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    # Node 1: HyDE
    def hyde_node(self, state: AgentState) -> AgentState:
        """Rewrite the question as a fake social media post for better retrieval."""
        chain = HYDE_PROMPT | self.llm | StrOutputParser()
        hypothetical = chain.invoke({"question": state["question"]})
        return {**state, "rewritten_query": hypothetical}

    # Node 2: Retriever
    def retrieve_node(self, state: AgentState) -> AgentState:
        """Retrieve the top 10 posts from Pinecone. Uses the HyDE query if available."""
        query = state["rewritten_query"] if state["rewritten_query"] else state["question"]
        docs = self.retriever.retrieve(query, k=10)
        return {**state, "retrieved_docs": docs}

    # Node 3: Answer (fine-tuned SmolLM2-360M) 
    def answer_node(self, state: AgentState) -> AgentState:
        """
        Format the retrieved posts and question, then pass them to SmolLM2.
        The prompt format matches what the model was trained on.
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
