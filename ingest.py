"""
One-time ingestion script — populates the Pinecone vector index.

Loads 50,000 English social media posts from the Exorde HuggingFace dataset,
embeds them with OpenAI's text-embedding-3-small, and upserts them into a
Pinecone serverless index. Run this once before starting the app.

Usage:
    python ingest.py

Requires OPENAI_API_KEY and PINECONE_API_KEY set in .env or the environment.
Runtime: ~30–60 minutes depending on API throughput.
"""

import os
import time
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from rag.preprocessing import load_chunks

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INDEX_NAME   = os.getenv("PINECONE_INDEX_NAME", "exorde-week1")
EMBED_MODEL  = "text-embedding-3-small"
EMBED_DIM    = 1536
CLOUD        = "aws"
REGION       = "us-east-1"
BATCH_SIZE   = 100


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
# Embedding
# ---------------------------------------------------------------------------

def embed_texts(client: OpenAI, texts: List[str], max_retries: int = 6) -> List[List[float]]:
    """Embed a batch of texts with exponential backoff on failure."""
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            wait = min(2 ** attempt, 30)
            print(f"[warn] Embedding failed (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(wait)
    raise RuntimeError("Embedding failed after max retries.")


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
    openai_key   = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")

    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    if not pinecone_key:
        raise EnvironmentError("PINECONE_API_KEY is not set.")

    client = OpenAI(api_key=openai_key)
    pc     = Pinecone(api_key=pinecone_key)

    # Step 1: Load and preprocess
    chunks = load_chunks()

    # Step 2: Create index
    index = get_or_create_index(pc)
    print("Index stats before upsert:", index.describe_index_stats())

    # Step 3: Embed and upsert
    print(f"\nEmbedding and upserting {len(chunks):,} chunks in batches of {BATCH_SIZE}...")
    for start in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Upserting"):
        batch = chunks[start:start + BATCH_SIZE]

        ids     = [c["chunk_id"] for c in batch]
        texts   = [c["text"] for c in batch]
        metas   = [safe_metadata(c["metadata"]) for c in batch]
        vectors = embed_texts(client, texts)

        index.upsert(vectors=[
            {"id": _id, "values": vec, "metadata": meta}
            for _id, vec, meta in zip(ids, vectors, metas)
        ])

    print("\nIngestion complete.")
    print("Index stats after upsert:", index.describe_index_stats())


if __name__ == "__main__":
    main()
