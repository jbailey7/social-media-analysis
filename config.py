"""
Central configuration for model identifiers.

All model names are defined here so that changing a model requires editing
exactly one file rather than hunting across the codebase.
"""

# OpenAI model used for HyDE query rewriting and answer generation (configs A/B)
CHAT_MODEL = "gpt-4o-mini"

# OpenAI embedding model used for ingestion and retrieval
EMBED_MODEL = "text-embedding-3-small"

# HuggingFace model ID for the answer generator (base and LoRA fine-tuned)
SMOLLM_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"

# HuggingFace model ID for the BGE embedding experiment (ingest_bge.py)
BGE_MODEL_ID = "BAAI/bge-large-en-v1.5"
