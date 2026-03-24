"""
Runs 7 test questions through 4 configurations and scores them with RAGAS.

Configs:
  A — gpt-4o-mini with no retrieval (baseline)
  B — Pinecone + gpt-4o-mini, no pipeline
  C — Full pipeline with base SmolLM2-360M
  D — Full pipeline with LoRA fine-tuned SmolLM2-360M

Config A is excluded from RAGAS since it has no retrieved context.
Results are saved to evaluation_results.json.
"""

import logging
import os
import json

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import CHAT_MODEL, EMBED_MODEL

load_dotenv()

logger = logging.getLogger(__name__)

# Test questions
QUESTIONS = [
    "What were people saying about cryptocurrency and Bitcoin in December 2024?",
    "What was public sentiment around AI tools like ChatGPT in December 2024?",
    "How did people react to the Syria conflict in December 2024?",
    "What were the trending political topics on social media in December 2024?",
    "How did people feel about the economy and inflation in December 2024?",
    "What were people saying about Elon Musk on social media in December 2024?",
    "What sports topics were trending on social media in December 2024?",
]

# Shared LLM
llm = ChatOpenAI(
    model=CHAT_MODEL,
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# Config A: Base LLM (no RAG)
def run_base_llm(question: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the question as best you can."),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})


# Config B: Basic RAG (retriever + gpt-4o-mini, no agents)
def run_basic_rag(question: str, retriever) -> dict:
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
    answer = chain.invoke({"context": context, "question": question})
    return {"answer": answer, "docs": docs}


# Config C & D: Advanced Agentic RAG
def run_advanced_rag(question: str, nodes) -> dict:
    """Run the full pipeline (HyDE → retrieve → answer) and return the result."""
    state = {
        "question": question,
        "rewritten_query": "",
        "retrieved_docs": [],
        "answer": "",
    }

    state = nodes.hyde_node(state)
    state = nodes.retrieve_node(state)
    state = nodes.answer_node(state)

    return {
        "answer": state["answer"],
        "rewritten_query": state.get("rewritten_query", ""),
        "docs": state["retrieved_docs"],
    }


# RAGAS scoring
def score_with_ragas(questions: list, answers: list, docs_list: list) -> list:
    """Score a list of (question, answer, docs) triples with RAGAS. Returns one score dict per question."""
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference
    from ragas.llms import llm_factory
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import OpenAIEmbeddings
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")

    # max_tokens bumped to 16000 so RAGAS doesn't truncate when scoring long answers.
    evaluator_llm = llm_factory(
        CHAT_MODEL,
        client=OpenAI(api_key=api_key),
        max_tokens=16000,
    )

    # Explicitly set the embedding model — without this RAGAS defaults to
    # text-embedding-ada-002, which this project's API key can't access.
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=api_key)
    )

    samples = [
        SingleTurnSample(
            user_input=q,
            response=a,
            retrieved_contexts=[doc.page_content for doc in docs],
        )
        for q, a, docs in zip(questions, answers, docs_list)
    ]
    dataset = EvaluationDataset(samples=samples)
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            # strictness=1 asks for 1 probe question instead of the default 3.
            # gpt-4o-mini only returns 1 at a time anyway, and the default setting
            # causes most scores to collapse to 0.0.
            ResponseRelevancy(strictness=1),
            LLMContextPrecisionWithoutReference(),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    return [dict(s) for s in result.scores]


def _avg_scores(scores: list) -> dict:
    """Average a list of per-question score dicts. Skips None values."""
    if not scores:
        return {}
    keys = scores[0].keys()
    return {
        k: round(
            sum(s[k] for s in scores if s.get(k) is not None) / len(scores), 4
        )
        for k in keys
    }


# Main evaluation loop
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 80)
    logger.info("Social Media Intelligence Agent — Evaluation")
    logger.info("=" * 80)

    logger.info("Initialising resources...")

    from rag.retriever import PineconeRetriever
    from models.smollm import load_model, load_base_model
    from agents.nodes import AgentNodes

    logger.info("Loading Pinecone retriever (may take ~2 min)...")
    retriever = PineconeRetriever()

    logger.info("Loading base SmolLM2-360M (no LoRA)...")
    base_model, base_tokenizer = load_base_model()

    logger.info("Loading fine-tuned SmolLM2-360M...")
    ft_model, ft_tokenizer = load_model()

    base_nodes = AgentNodes(retriever, base_model, base_tokenizer)
    ft_nodes   = AgentNodes(retriever, ft_model,   ft_tokenizer)

    logger.info("All resources loaded. Starting evaluation.")
    logger.info("=" * 80)

    results = []

    # Accumulators for RAGAS (configs B, C, D only)
    ragas_inputs = {
        "b": {"questions": [], "answers": [], "docs": []},
        "c": {"questions": [], "answers": [], "docs": []},
        "d": {"questions": [], "answers": [], "docs": []},
    }

    for i, question in enumerate(QUESTIONS, 1):
        logger.info("Q%d/%d: %s", i, len(QUESTIONS), question)
        logger.info("-" * 80)

        entry = {"question": question}

        # Config A
        logger.info("Running A (Base LLM)...")
        entry["a_base_llm"] = run_base_llm(question)

        # Config B
        logger.info("Running B (Basic RAG)...")
        b = run_basic_rag(question, retriever)
        entry["b_basic_rag"] = b["answer"]
        ragas_inputs["b"]["questions"].append(question)
        ragas_inputs["b"]["answers"].append(b["answer"])
        ragas_inputs["b"]["docs"].append(b["docs"])

        # Config C
        logger.info("Running C (Advanced RAG, base SmolLM2)...")
        c = run_advanced_rag(question, base_nodes)
        entry["c_advanced_base"] = c["answer"]
        entry["c_hyde_query"] = c["rewritten_query"]
        ragas_inputs["c"]["questions"].append(question)
        ragas_inputs["c"]["answers"].append(c["answer"])
        ragas_inputs["c"]["docs"].append(c["docs"])

        # Config D
        logger.info("Running D (Advanced RAG, fine-tuned SmolLM2)...")
        d = run_advanced_rag(question, ft_nodes)
        entry["d_advanced_finetuned"] = d["answer"]
        entry["d_hyde_query"] = d["rewritten_query"]
        ragas_inputs["d"]["questions"].append(question)
        ragas_inputs["d"]["answers"].append(d["answer"])
        ragas_inputs["d"]["docs"].append(d["docs"])

        results.append(entry)

        logger.info("A: %s...", entry['a_base_llm'][:120])
        logger.info("B: %s...", entry['b_basic_rag'][:120])
        logger.info("C: %s...", entry['c_advanced_base'][:120])
        logger.info("D: %s...", entry['d_advanced_finetuned'][:120])

    # RAGAS scoring
    logger.info("=" * 80)
    logger.info("Running RAGAS evaluation (B, C, D)...")
    logger.info("Config A is excluded — no retrieved context to evaluate.")
    logger.info("=" * 80)

    config_labels = {
        "b": "B — Basic RAG",
        "c": "C — Advanced RAG (base)",
        "d": "D — Advanced RAG (fine-tuned)",
    }

    ragas_summary = {}
    for key, label in config_labels.items():
        logger.info("Scoring %s...", label)
        scores = score_with_ragas(
            ragas_inputs[key]["questions"],
            ragas_inputs[key]["answers"],
            ragas_inputs[key]["docs"],
        )
        ragas_summary[key] = _avg_scores(scores)
        for i, score in enumerate(scores):
            results[i][f"{key}_ragas"] = score

    # Print summary table
    logger.info("=" * 80)
    logger.info("RAGAS Summary (averages across all questions)")
    logger.info("=" * 80)

    metric_names = list(ragas_summary["b"].keys())
    col_w = 24
    header = f"{'Config':<34}" + "".join(f"{m:<{col_w}}" for m in metric_names)
    logger.info(header)
    logger.info("-" * len(header))
    for key, label in config_labels.items():
        row = f"{label:<34}" + "".join(
            f"{ragas_summary[key].get(m, 'N/A'):<{col_w}}" for m in metric_names
        )
        logger.info(row)

    # Save results
    output = {
        "per_question": results,
        "ragas_summary": ragas_summary,
    }
    out_path = "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("=" * 80)
    logger.info("Done. Results saved to %s", out_path)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
