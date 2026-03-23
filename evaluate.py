"""
Evaluation script for the Social Media Intelligence Agent.

Runs 7 test questions through 4 configurations and saves results to
evaluation_results.json.

Configurations:
  A — Base LLM          — gpt-4o-mini with no retrieval
  B — Basic RAG         — Pinecone retriever + gpt-4o-mini
  C — Advanced RAG      — Full agentic pipeline with base SmolLM2-360M
  D — Advanced RAG (FT) — Full agentic pipeline with LoRA fine-tuned SmolLM2-360M

Usage:
  python evaluate.py
"""

import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ---------------------------------------------------------------------------
# Test questions
# ---------------------------------------------------------------------------

QUESTIONS = [
    "What were people saying about cryptocurrency and Bitcoin in December 2024?",
    "What was public sentiment around AI tools like ChatGPT in December 2024?",
    "How did people react to the Syria conflict in December 2024?",
    "What were the trending political topics on social media in December 2024?",
    "How did people feel about the economy and inflation in December 2024?",
    "What were people saying about Elon Musk on social media in December 2024?",
    "What sports topics were trending on social media in December 2024?",
]

# ---------------------------------------------------------------------------
# Shared LLM
# ---------------------------------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# ---------------------------------------------------------------------------
# Config A: Base LLM (no RAG)
# ---------------------------------------------------------------------------

def run_base_llm(question: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the question as best you can."),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})


# ---------------------------------------------------------------------------
# Config B: Basic RAG (retriever + gpt-4o-mini, no agents)
# ---------------------------------------------------------------------------

def run_basic_rag(question: str, retriever) -> str:
    docs = retriever.retrieve(question, k=10)
    context = "\n\n".join([
        f"Post [{doc.metadata.get('primary_theme', 'General')}]: {doc.page_content}"
        for doc in docs
    ])
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful assistant that answers questions about social media trends "
            "in December 2024. Use the retrieved posts below to inform your answer.\n\n"
            "Retrieved posts:\n{context}"
        )),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


# ---------------------------------------------------------------------------
# Config C & D: Advanced Agentic RAG
# ---------------------------------------------------------------------------

def run_advanced_rag(question: str, nodes) -> dict:
    """
    Runs router → optional HyDE → retrieve → SmolLM2 answer using the
    provided AgentNodes instance. Returns a dict with 'answer' and trace fields.
    """
    state = {
        "question": question,
        "rewritten_query": "",
        "retrieved_docs": [],
        "answer": "",
        "use_hyde": False,
        "router_reason": "",
    }

    state = nodes.router_node(state)
    if state["use_hyde"]:
        state = nodes.hyde_node(state)
    state = nodes.retrieve_node(state)
    state = nodes.answer_node(state)

    return {
        "answer": state["answer"],
        "use_hyde": state["use_hyde"],
        "router_reason": state["router_reason"],
        "rewritten_query": state.get("rewritten_query", ""),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Social Media Intelligence Agent — Evaluation")
    print("=" * 80)

    # Initialise all resources once before the question loop
    print("\nInitialising resources...")

    from rag.retriever import PineconeRetriever
    from models.smollm import load_model, load_base_model
    from agents.nodes import AgentNodes

    print("  Loading Pinecone retriever (may take ~2 min)...")
    retriever = PineconeRetriever()

    print("  Loading base SmolLM2-360M (no LoRA)...")
    base_model, base_tokenizer = load_base_model()

    print("  Loading fine-tuned SmolLM2-360M...")
    ft_model, ft_tokenizer = load_model()

    base_nodes = AgentNodes(retriever, base_model, base_tokenizer)
    ft_nodes   = AgentNodes(retriever, ft_model,   ft_tokenizer)

    print("\nAll resources loaded. Starting evaluation.\n")
    print("=" * 80)

    results = []

    for i, question in enumerate(QUESTIONS, 1):
        print(f"\nQ{i}/{len(QUESTIONS)}: {question}")
        print("-" * 80)

        entry = {"question": question}

        # Config A
        print("  Running A (Base LLM)...")
        entry["a_base_llm"] = run_base_llm(question)

        # Config B
        print("  Running B (Basic RAG)...")
        entry["b_basic_rag"] = run_basic_rag(question, retriever)

        # Config C
        print("  Running C (Advanced RAG, base SmolLM2)...")
        c = run_advanced_rag(question, base_nodes)
        entry["c_advanced_base"] = c["answer"]
        entry["c_trace"] = {k: v for k, v in c.items() if k != "answer"}

        # Config D
        print("  Running D (Advanced RAG, fine-tuned SmolLM2)...")
        d = run_advanced_rag(question, ft_nodes)
        entry["d_advanced_finetuned"] = d["answer"]
        entry["d_trace"] = {k: v for k, v in d.items() if k != "answer"}

        results.append(entry)

        # Quick preview
        print(f"\n  A: {entry['a_base_llm'][:120]}...")
        print(f"  B: {entry['b_basic_rag'][:120]}...")
        print(f"  C: {entry['c_advanced_base'][:120]}...")
        print(f"  D: {entry['d_advanced_finetuned'][:120]}...")

    # Save results
    out_path = "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Done. Results saved to {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
