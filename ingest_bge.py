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

import os
import time

import numpy as np
import torch
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag.preprocessing import load_chunks

load_dotenv()

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
        print(f"Creating Pinecone index '{INDEX_NAME}' (dim={EMBED_DIM}, metric=cosine)...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(2)
        print("Index created and ready.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")

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
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        raise EnvironmentError("PINECONE_API_KEY is not set.")

    # Auto-select device: CUDA GPU if available, otherwise CPU.
    # MPS (Apple Silicon) is intentionally excluded due to memory limitations.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Loading BAAI/bge-large-en-v1.5...")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
    print(f"Model loaded. Embedding dim: {EMBED_DIM}")

    pc = Pinecone(api_key=pinecone_key)

    # Step 1: Load and preprocess
    chunks = load_chunks()

    # Step 2: Create index
    index = get_or_create_index(pc)
    print("Index stats before upsert:", index.describe_index_stats())

    # Step 3: Embed and upsert
    print(f"\nEmbedding and upserting {len(chunks):,} chunks in batches of {BATCH_SIZE}...")
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

    print("\nIngestion complete.")
    print("Index stats after upsert:", index.describe_index_stats())


if __name__ == "__main__":
    main()
