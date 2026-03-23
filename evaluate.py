"""
Evaluation script for the Social Media Intelligence Agent.

Runs 7 test questions through 4 configurations and saves results to
evaluation_results.json.

Configurations:
  a. Base LLM          — gpt-4o-mini with no retrieval
  b. Basic RAG         — Pinecone retriever + gpt-4o-mini
  c. Advanced RAG      — Full agentic pipeline (router + HyDE + retriever) with base SmolLM2
  d. Advanced RAG (FT) — Full agentic pipeline with LoRA fine-tuned SmolLM2

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
# Resource caches (load once, reuse across questions)
# ---------------------------------------------------------------------------

_retriever = None
_ft_model_cache = None
_base_model_cache = None


def get_retriever():
    global _retriever
    if _retriever is None:
        print("  [init] Building Pinecone retriever (may take ~2 min)...")
        from rag.retriever import PineconeRetriever
        _retriever = PineconeRetriever()
        print("  [init] Retriever ready.")
    return _retriever


def get_ft_model():
    global _ft_model_cache
    if _ft_model_cache is None:
        print("  [init] Loading fine-tuned SmolLM2-360M...")
        from models.smollm import load_model
        _ft_model_cache = load_model()
        print("  [init] Fine-tuned model ready.")
    return _ft_model_cache


def get_base_model():
    global _base_model_cache
    if _base_model_cache is None:
        print("  [init] Loading base SmolLM2-360M (no LoRA)...")
        from models.smollm import load_base_model
        _base_model_cache = load_base_model()
        print("  [init] Base model ready.")
    return _base_model_cache


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

def run_basic_rag(question: str) -> str:
    retriever = get_retriever()
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

def _build_dialogue(docs, question: str) -> str:
    """Format retrieved docs and question as context for SmolLM2."""
    lines = []
    for i, doc in enumerate(docs[:7], start=1):
        theme = doc.metadata.get("primary_theme", "General")
        lines.append(f"Post {i} [{theme}]: {doc.page_content}")
    lines.append(f"\nQuestion: {question}")
    return "\n".join(lines)


def run_advanced_rag(question: str, use_fine_tuned: bool) -> dict:
    """
    Runs router → optional HyDE → retrieve → SmolLM2 answer.
    Returns a dict with 'answer' and agent trace fields.
    """
    from agents.nodes import router_node, hyde_node, retrieve_node

    state = {
        "question": question,
        "rewritten_query": "",
        "retrieved_docs": [],
        "answer": "",
        "use_hyde": False,
        "router_reason": "",
    }

    state = router_node(state)
    if state["use_hyde"]:
        state = hyde_node(state)
    state = retrieve_node(state)

    dialogue = _build_dialogue(state["retrieved_docs"], question)

    from models.smollm import generate_summary
    if use_fine_tuned:
        model, tokenizer = get_ft_model()
    else:
        model, tokenizer = get_base_model()

    answer = generate_summary(model, tokenizer, dialogue, max_new_tokens=150)

    return {
        "answer": answer,
        "use_hyde": state["use_hyde"],
        "router_reason": state["router_reason"],
        "rewritten_query": state.get("rewritten_query", ""),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Stitching Project — Evaluation")
    print("=" * 80)

    # Pre-load all resources before the question loop
    print("\nPre-loading resources...")
    get_retriever()
    get_ft_model()
    get_base_model()
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
        entry["b_basic_rag"] = run_basic_rag(question)

        # Config C
        print("  Running C (Advanced RAG, base SmolLM2)...")
        c = run_advanced_rag(question, use_fine_tuned=False)
        entry["c_advanced_base"] = c["answer"]
        entry["c_trace"] = {k: v for k, v in c.items() if k != "answer"}

        # Config D
        print("  Running D (Advanced RAG, fine-tuned SmolLM2)...")
        d = run_advanced_rag(question, use_fine_tuned=True)
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
