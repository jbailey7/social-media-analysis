"""
Ingestion script using theme-grouped chunking.

Rather than treating each social media post as an individual chunk, this script
groups posts by their primary_theme field and concatenates them into larger chunks.
Each chunk represents a coherent thematic unit, giving the embedding model richer
context and reducing the total number of vectors in the index.

Index name:  exorde-chunked-theme
Embedding:   text-embedding-3-small (1536 dimensions)
Chunk size:  5 posts per chunk (configurable via GROUP_SIZE)

Usage:
    python ingest_theme_grouped.py

Requires OPENAI_API_KEY and PINECONE_API_KEY set in .env or the environment.
Runtime: ~30–60 minutes depending on API throughput.
"""

import logging
import os
import time
from collections import defaultdict
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from rag.preprocessing import load_chunks
from config import EMBED_MODEL

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INDEX_NAME  = os.getenv("PINECONE_THEME_INDEX_NAME", "exorde-chunked-theme")
EMBED_DIM   = 1536
CLOUD       = "aws"
REGION      = "us-east-1"
BATCH_SIZE  = 100
GROUP_SIZE  = 5   # number of posts to combine per theme-chunk


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def build_theme_chunks(chunks: list, group_size: int = GROUP_SIZE) -> list:
    """
    Group posts by primary_theme and concatenate into larger chunks.

    Posts within each theme are grouped sequentially in groups of group_size.
    The combined text uses ' | ' as a separator between posts.

    chunk_id format: theme_<theme>_<group_start_index>
    """
    by_theme = defaultdict(list)
    for c in chunks:
        theme = c["metadata"].get("primary_theme") or "General"
        by_theme[theme].append(c)

    theme_chunks = []
    for theme, posts in by_theme.items():
        for i in range(0, len(posts), group_size):
            group = posts[i:i + group_size]
            combined_text = " | ".join(p["text"] for p in group)
            theme_chunks.append({
                "chunk_id": f"theme_{theme}_{i}",
                "text":     combined_text,
                "metadata": {
                    "primary_theme": theme,
                    "n_posts":       len(group),
                },
            })

    logger.info("Built %s theme-grouped chunks from %s posts (%d unique themes, group_size=%d)",
                f"{len(theme_chunks):,}", f"{len(chunks):,}", len(by_theme), group_size)
    return theme_chunks


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
            logger.warning("Embedding failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
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

    # Step 1: Load individual posts
    chunks = load_chunks()

    # Step 2: Group into theme chunks
    theme_chunks = build_theme_chunks(chunks, group_size=GROUP_SIZE)

    # Step 3: Create index
    index = get_or_create_index(pc)
    logger.info("Index stats before upsert: %s", index.describe_index_stats())

    # Step 4: Embed and upsert
    logger.info("Embedding and upserting %s chunks in batches of %d...", f"{len(theme_chunks):,}", BATCH_SIZE)
    for start in tqdm(range(0, len(theme_chunks), BATCH_SIZE), desc="Upserting"):
        batch = theme_chunks[start:start + BATCH_SIZE]

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
