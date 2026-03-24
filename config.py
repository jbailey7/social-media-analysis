"""
All model names in one place. Change a model here and it updates everywhere.
"""

# Used for HyDE query rewriting and answer generation (configs A/B)
CHAT_MODEL = "gpt-4o-mini"

# Used for ingestion and retrieval
EMBED_MODEL = "text-embedding-3-small"

# The small local model used as the answer generator
SMOLLM_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"

# Only used in the BGE embedding experiment (ingest_bge.py)
BGE_MODEL_ID = "BAAI/bge-large-en-v1.5"
