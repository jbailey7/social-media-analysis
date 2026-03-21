# Social Media Intelligence Agent

A multi-agent RAG system built with LangGraph that answers natural language questions
about social media discussions from December 2024. The system combines semantic vector
search, LLM-driven query routing, HyDE query rewriting, and a LoRA fine-tuned language
model into a single agentic pipeline.

## System Overview

### Agent Pipeline

Queries are processed by a four-node LangGraph graph:

| Node | Model | Role |
|---|---|---|
| `router_node` | gpt-4o-mini | Decides whether HyDE query rewriting should be used |
| `hyde_node` | gpt-4o-mini | Rewrites the question as a hypothetical social media post |
| `retrieve_node` | Pinecone | Retrieves top-10 relevant posts from the vector index |
| `answer_node` | SmolLM2-360M (LoRA fine-tuned) | Summarizes retrieved posts into a final answer |

The router uses no hard-coded logic — gpt-4o-mini decides whether HyDE is beneficial
for each query and routes accordingly. Sentiment and opinion questions are typically
routed through HyDE; specific factual or named-entity queries go directly to retrieval.

![Agent graph](graph.png)

### Vector Index

The Pinecone index holds 50,000 English social media posts from the
[Exorde December 2024 dataset](https://huggingface.co/datasets/Exorde/exorde-social-media-december-2024-week1),
embedded with OpenAI's `text-embedding-3-small` (1536 dimensions, cosine similarity).
Pinecone stores only lightweight metadata; full post text is resolved at query time
from a local lookup dict rebuilt from the HuggingFace dataset.

### Fine-tuned Answer Model

`SmolLM2-360M-Instruct` (360M parameter causal LM) is fine-tuned using LoRA (PEFT)
on the [SAMSum dialogue summarization dataset](https://huggingface.co/datasets/knkarthick/samsum).
Fine-tuning targets the `q_proj` and `v_proj` attention layers (`r=8, alpha=32`) and
trains only 0.23% of total parameters (~819k of 362M). The adapter weights are
committed to this repository under `fine_tuned_model/`.

## Project Structure

```
social-media-analysis/
├── app.py                      # Streamlit front-end
├── evaluate.py                 # Evaluation script (4 configs x 7 questions)
├── ingest.py                   # Populates Pinecone index using text-embedding-3-small
├── ingest_bge.py               # Populates Pinecone index using BAAI/bge-large-en-v1.5
├── ingest_theme_grouped.py     # Populates Pinecone index using theme-grouped chunking
├── train.py                    # LoRA fine-tuning script for the answer model
├── agents/
│   ├── state.py                # AgentState TypedDict
│   ├── prompts.py              # Router and HyDE prompt templates
│   ├── nodes.py                # Node functions (router, hyde, retrieve, answer)
│   └── graph.py                # LangGraph graph definition and compilation
├── rag/
│   ├── preprocessing.py        # Shared Exorde dataset loading and cleaning
│   └── retriever.py            # Pinecone client + chunk_lookup + retrieve function
├── models/
│   └── smollm.py               # SmolLM2-360M loader and generate function
├── notebooks/
│   └── rag_experiments.ipynb   # RAG configuration experiments and comparisons
├── fine_tuned_model/           # LoRA adapter weights (~2 MB, committed)
├── requirements.txt
└── .env.example
```

## Prerequisites

### Python Version

**Python 3.11 or 3.12 is required.** PyTorch does not have builds for Python 3.13+.

```bash
python --version  # must be 3.11.x or 3.12.x
```

If your default Python is 3.13, create a virtual environment using an older version:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### API Keys

This project requires:
- **OpenAI API key** — for embeddings (`text-embedding-3-small`) and the router/HyDE nodes (`gpt-4o-mini`)
- **Pinecone API key** — for vector storage and retrieval

### Pinecone Vector Index

The app queries the Pinecone index specified by `PINECONE_INDEX_NAME` in your `.env` file. The index must exist and be populated before running the app. See **Populating the Index** below.

Index settings:
- **Dimensions:** 1536
- **Metric:** cosine
- **Cloud/Region:** AWS `us-east-1`

### Fine-tuned Model Weights

The LoRA adapter weights are committed under `fine_tuned_model/`. The base model
(`HuggingFaceTB/SmolLM2-360M-Instruct`, ~750 MB) is downloaded automatically from
HuggingFace on first run and cached locally.

## Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd social-media-analysis

# 2. Create and activate a virtual environment (Python 3.11 or 3.12)
python3.11 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY and PINECONE_API_KEY
```

## Populating the Index

Three ingestion scripts are available, each building a different Pinecone index.
The primary index is required to run the app; the others are only needed for the
RAG experiments notebook.

| Script | Index env var | Embedding model | Purpose |
|---|---|---|---|
| `ingest.py` | `PINECONE_INDEX_NAME` | `text-embedding-3-small` | Primary index used by the app |
| `ingest_bge.py` | `PINECONE_BGE_INDEX_NAME` | `BAAI/bge-large-en-v1.5` (local) | Experiment 4 — embedding model comparison |
| `ingest_theme_grouped.py` | `PINECONE_THEME_INDEX_NAME` | `text-embedding-3-small` | Experiment 5 — chunking strategy comparison |

Run the primary index first:

```bash
python ingest.py
```

Runtime is approximately 30–60 minutes depending on API throughput. The index only
needs to be populated once; subsequent app runs query the existing index.

`ingest_bge.py` embeds all 50,000 posts locally using `BAAI/bge-large-en-v1.5` and
requires no API calls. It automatically uses a CUDA GPU if available, otherwise falls
back to CPU. A GPU is strongly recommended — CPU inference on this model is very slow.

## Running the Web App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. On first launch the app will:

1. Download and cache the Exorde dataset, then rebuild the text lookup (~2 min)
2. Download the SmolLM2-360M base model from HuggingFace (first time only)
3. Load the LoRA adapter from `fine_tuned_model/`
4. Compile the LangGraph graph

Subsequent runs skip steps 1–3 thanks to caching.

## RAG Experiments

`notebooks/rag_experiments.ipynb` systematically evaluates different RAG configurations
to find the best setup for this system. Each experiment isolates one variable, states
a hypothesis, measures retrieval quality across 7 test questions, and records findings.

| Experiment | Variable | Configurations |
|---|---|---|
| 1 | Top-k | k = 5, 10, 20 |
| 2 | Query enhancement | None, always HyDE, conditional HyDE, multi-query |
| 3 | Reranking | None, cross-encoder, LLM-based |
| 4 | Embedding model | `text-embedding-3-small`, `BAAI/bge-large-en-v1.5` |
| 5 | Chunking strategy | Individual posts, theme-grouped |

Experiments 1–3 require only the primary Pinecone index. Experiments 4 and 5 require
their respective indexes to be built first (see **Populating the Index** above).

To run the notebook:

```bash
cd notebooks
jupyter notebook rag_experiments.ipynb
```

Results from all experiments are saved to `rag_experiment_results.json`.

## Running the Evaluation

```bash
python evaluate.py
```

Runs 7 test questions through 4 configurations to measure the contribution of each
system component:

| Config | Setup |
|---|---|
| **A — Base LLM** | gpt-4o-mini with no retrieval |
| **B — Basic RAG** | Pinecone retriever + gpt-4o-mini, no agents |
| **C — Advanced RAG (base)** | Full agentic pipeline with untuned SmolLM2-360M |
| **D — Advanced RAG (fine-tuned)** | Full agentic pipeline with LoRA fine-tuned SmolLM2-360M |

Results are saved to `evaluation_results.json`.

## Re-training the Answer Model

The committed adapter weights can be reproduced by running:

```bash
python train.py
```

This fine-tunes `SmolLM2-360M-Instruct` on 1,000 examples from the SAMSum dataset
using LoRA, and saves the adapter weights to `fine_tuned_model/`. A GPU is recommended
(~4 minutes on CUDA; significantly longer on CPU). No API keys are required.

## Sample Questions

- "What were people saying about cryptocurrency and Bitcoin in December 2024?"
- "What happened with the Syria conflict in early December 2024?"
- "How did people feel about AI tools and ChatGPT on social media?"
- "What were the trending topics on social media during the first week of December 2024?"
- "What was the public reaction to Joe Biden on social media in December 2024?"

## Roadmap

### Immediate (GPU machine)
- [ ] Rebuild `exorde-week1` index (`python ingest.py`) — current index is corrupted
- [ ] Run `ingest_bge.py` on GPU to build `exorde-embed-bge`
- [ ] Run `ingest_theme_grouped.py` to build `exorde-chunked-theme`
- [ ] Complete notebook experiments 4 and 5
- [ ] Fill in findings and conclusions in the notebook
- [ ] Update `rag/retriever.py` with the winning configuration from experiments

### Upcoming
- [ ] Address SAMSum fine-tuning mismatch in answer model
- [ ] Add tests
- [ ] Refactor global cache pattern in `agents/nodes.py`

## Notes

- Do not commit `.env` or any dataset files
- `evaluation_results.json` and `rag_experiment_results.json` are generated locally and do not need to be committed
- API keys must never be left in source code
