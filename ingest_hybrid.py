"""
One-time ingestion script — populates the hybrid Pinecone index.

Hybrid search combines dense vectors (OpenAI text-embedding-3-small) with
sparse vectors (BM25) to capture both semantic similarity and keyword overlap.
Pinecone requires 'dotproduct' as the metric for hybrid indexes.

BM25 params are fitted on the full corpus and saved to bm25_params.json so
the HybridRetriever can produce matching sparse query vectors at runtime
without re-fitting.

Usage:
    python ingest_hybrid.py

Requires OPENAI_API_KEY and PINECONE_API_KEY set in .env or the environment.
Runtime: ~30–60 minutes depending on API throughput.
"""

import json
import logging
import os
import time
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from tqdm import tqdm

from rag.preprocessing import load_chunks
from ingest import safe_metadata, embed_texts
from config import EMBED_MODEL

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INDEX_NAME    = os.getenv("PINECONE_HYBRID_INDEX_NAME", "exorde-hybrid")
EMBED_DIM     = 1536
CLOUD         = "aws"
REGION        = "us-east-1"
BATCH_SIZE    = 100
BM25_PARAMS_PATH = "bm25_params.json"


# ---------------------------------------------------------------------------
# Pinecone index setup
# ---------------------------------------------------------------------------

def get_or_create_index(pc: Pinecone):
    """Create a dotproduct index for hybrid search if it doesn't exist."""
    existing = [idx["name"] for idx in pc.list_indexes()]

    if INDEX_NAME not in existing:
        logger.info("Creating Pinecone index '%s' (dim=%d, metric=dotproduct)...", INDEX_NAME, EMBED_DIM)
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="dotproduct",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(2)
        logger.info("Index created and ready.")
    else:
        logger.info("Index '%s' already exists.", INDEX_NAME)

    return pc.Index(INDEX_NAME)


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

def fit_and_save_bm25(texts: List[str]) -> BM25Encoder:
    """Fit BM25 on the full corpus and save params for later query-time use."""
    logger.info("Fitting BM25 encoder on corpus...")
    bm25 = BM25Encoder()
    bm25.fit(texts)
    bm25.dump(BM25_PARAMS_PATH)
    logger.info("BM25 params saved to %s", BM25_PARAMS_PATH)
    return bm25


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

    # Step 1: Load and preprocess
    chunks = load_chunks()
    texts  = [c["text"] for c in chunks]

    # Step 2: Fit BM25 and save params
    bm25 = fit_and_save_bm25(texts)

    # Step 3: Create index
    index = get_or_create_index(pc)
    logger.info("Index stats before upsert: %s", index.describe_index_stats())

    # Step 4: Embed + encode sparse + upsert
    # Vectors are stored unscaled — alpha blending happens at query time,
    # which allows different alpha values to be tested without re-ingesting.
    logger.info("Embedding and upserting %s chunks in batches of %d...", f"{len(chunks):,}", BATCH_SIZE)
    total_skipped = 0
    for start in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Upserting"):
        batch = chunks[start:start + BATCH_SIZE]

        ids         = [c["chunk_id"] for c in batch]
        batch_texts = [c["text"] for c in batch]
        metas       = [safe_metadata(c["metadata"]) for c in batch]

        dense_vecs  = embed_texts(client, batch_texts)
        sparse_vecs = bm25.encode_documents(batch_texts)

        # Skip any document whose sparse vector is empty (BM25 found no
        # known tokens — happens with very short or heavily-punctuated posts).
        vectors = []
        skipped = 0
        for _id, dense, sparse, meta in zip(ids, dense_vecs, sparse_vecs, metas):
            if not sparse.get("indices"):
                skipped += 1
                continue
            vectors.append({
                "id":            _id,
                "values":        dense,
                "sparse_values": sparse,
                "metadata":      meta,
            })

        if skipped:
            total_skipped += skipped
            tqdm.write(f"  ⚠ Skipped {skipped} doc(s) with empty sparse vectors in this batch")

        if vectors:
            index.upsert(vectors=vectors)

    logger.info("Ingestion complete. Skipped %s docs with empty sparse vectors.", f"{total_skipped:,}")
    logger.info("Index stats after upsert: %s", index.describe_index_stats())


if __name__ == "__main__":
    main()
