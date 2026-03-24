"""
One-time setup script — builds the main Pinecone index.

Loads 50k English posts from the Exorde dataset, embeds them with
text-embedding-3-small, and upserts everything into Pinecone.
Run this once before starting the app. Takes 30–60 minutes.
"""

import logging
import os
import time
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from rag.preprocessing import load_chunks
from config import EMBED_MODEL

load_dotenv()

logger = logging.getLogger(__name__)

# Config
INDEX_NAME   = os.getenv("PINECONE_INDEX_NAME", "exorde-week1")
EMBED_DIM    = 1536
CLOUD        = "aws"
REGION       = "us-east-1"
BATCH_SIZE   = 100


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


# Embedding
def embed_texts(client: OpenAI, texts: List[str], max_retries: int = 6) -> List[List[float]]:
    """Embed a batch of texts. Retries with exponential backoff if the API call fails."""
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            wait = min(2 ** attempt, 30)
            logger.warning("Embedding failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
            time.sleep(wait)
    raise RuntimeError("Embedding failed after max retries.")


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


# Main
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

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
    logger.info("Index stats before upsert: %s", index.describe_index_stats())

    # Step 3: Embed and upsert
    logger.info("Embedding and upserting %s chunks in batches of %d...", f"{len(chunks):,}", BATCH_SIZE)
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

    logger.info("Ingestion complete.")
    logger.info("Index stats after upsert: %s", index.describe_index_stats())


if __name__ == "__main__":
    main()
