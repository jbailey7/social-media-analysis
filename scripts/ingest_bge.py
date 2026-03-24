"""
Builds the BGE embedding index for Experiment 4.

Uses BAAI/bge-large-en-v1.5 (local, no API calls) instead of text-embedding-3-small
to see if a different embedding model improves retrieval. Needs only a Pinecone key.
Takes 60–90 minutes on CPU; much faster with a GPU.
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

# Config
INDEX_NAME  = os.getenv("PINECONE_BGE_INDEX_NAME", "exorde-embed-bge")
EMBED_DIM   = 1024   # BAAI/bge-large-en-v1.5 output dimension
CLOUD       = "aws"
REGION      = "us-east-1"
BATCH_SIZE  = 64     # smaller batch size than OpenAI due to local inference


# Pinecone index setup
def get_or_create_index(pc: Pinecone):
    """Return the Pinecone index, creating it first if it doesn't exist."""
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


# Metadata safety
def safe_metadata(meta: dict) -> dict:
    """Pinecone only accepts strings, numbers, and bools — drop anything else."""
    out = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        raise EnvironmentError("PINECONE_API_KEY is not set.")

    # Use CUDA if available. MPS (Apple Silicon) is skipped due to memory issues.
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

        # BGE recommends normalising for cosine similarity
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
