"""
Ingestion script using BAAI/bge-large-en-v1.5 embeddings.

Builds a Pinecone index from the Exorde dataset using a local open-source
embedding model rather than the OpenAI API. This eliminates the per-query
embedding cost and removes the OpenAI dependency from the retrieval path.

Index name:  exorde-embed-bge  (1024 dimensions)
Embedding:   BAAI/bge-large-en-v1.5 via sentence-transformers

Usage:
    python ingest_bge.py

Requires only PINECONE_API_KEY — no OpenAI key needed.
Runtime: ~60–90 minutes on CPU; faster on GPU.
"""

import logging
import os
import time

import numpy as np
import torch
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag.preprocessing import load_chunks
from config import BGE_MODEL_ID

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INDEX_NAME  = os.getenv("PINECONE_BGE_INDEX_NAME", "exorde-embed-bge")
EMBED_DIM   = 1024   # BAAI/bge-large-en-v1.5 output dimension
CLOUD       = "aws"
REGION      = "us-east-1"
BATCH_SIZE  = 64     # smaller batch size than OpenAI due to local inference


# ---------------------------------------------------------------------------
# Pinecone index setup
# ---------------------------------------------------------------------------

def get_or_create_index(pc: Pinecone):
    """Create the Pinecone index if it doesn't exist, then return it."""
    existing = [idx["name"] for idx in pc.list_indexes()]

    if INDEX_NAME not in existing:
        logger.info("Creating Pinecone index '%s' (dim=%d, metric=cosine)...", INDEX_NAME, EMBED_DIM)
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(2)
        logger.info("Index created and ready.")
    else:
        logger.info("Index '%s' already exists.", INDEX_NAME)

    return pc.Index(INDEX_NAME)


# ---------------------------------------------------------------------------
# Metadata safety
# ---------------------------------------------------------------------------

def safe_metadata(meta: dict) -> dict:
    """Strip non-serializable values so Pinecone accepts the metadata."""
    out = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        raise EnvironmentError("PINECONE_API_KEY is not set.")

    # Auto-select device: CUDA GPU if available, otherwise CPU.
    # MPS (Apple Silicon) is intentionally excluded due to memory limitations.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("Using device: %s", device)

    logger.info("Loading %s...", BGE_MODEL_ID)
    model = SentenceTransformer(BGE_MODEL_ID, device=device)
    logger.info("Model loaded. Embedding dim: %d", EMBED_DIM)

    pc = Pinecone(api_key=pinecone_key)

    # Step 1: Load and preprocess
    chunks = load_chunks()

    # Step 2: Create index
    index = get_or_create_index(pc)
    logger.info("Index stats before upsert: %s", index.describe_index_stats())

    # Step 3: Embed and upsert
    logger.info("Embedding and upserting %s chunks in batches of %d...", f"{len(chunks):,}", BATCH_SIZE)
    for start in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Upserting"):
        batch = chunks[start:start + BATCH_SIZE]

        ids   = [c["chunk_id"] for c in batch]
        texts = [c["text"] for c in batch]
        metas = [safe_metadata(c["metadata"]) for c in batch]

        # BGE recommends normalising embeddings for cosine similarity
        embeddings = model.encode(texts, normalize_embeddings=True)
        vectors    = embeddings.tolist()

        index.upsert(vectors=[
            {"id": _id, "values": vec, "metadata": meta}
            for _id, vec, meta in zip(ids, vectors, metas)
        ])

    logger.info("Ingestion complete.")
    logger.info("Index stats after upsert: %s", index.describe_index_stats())


if __name__ == "__main__":
    main()
