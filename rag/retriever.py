"""
Pinecone retrievers.

PineconeRetriever  — dense-only retrieval using text-embedding-3-small.
HybridRetriever    — dense + sparse (BM25) retrieval with alpha blending.

Both classes share the same chunk_lookup strategy: full post text is stored
locally and resolved by chunk ID, keeping Pinecone metadata lightweight.

Hybrid search requires a Pinecone index created with metric='dotproduct' and
BM25 params pre-fitted by ingest_hybrid.py (saved to bm25_params.json).
Alpha controls the blend: 1.0 = pure dense, 0.0 = pure sparse.
"""

import os
from typing import List

from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from rag.preprocessing import load_chunks

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
INDEX_NAME  = os.getenv("PINECONE_INDEX_NAME", "exorde-week1")


class PineconeRetriever:
    """Wraps Pinecone + local chunk_lookup for text-resolved retrieval."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=EMBED_MODEL,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index        = pc.Index(INDEX_NAME)
        self.chunk_lookup = {c["chunk_id"]: c for c in load_chunks()}

    def retrieve(self, query: str, k: int = 10) -> List[Document]:
        """
        Embed query, search Pinecone, resolve full text from chunk_lookup.
        Returns a list of LangChain Document objects.
        """
        query_vec = self.embeddings.embed_query(query)
        results   = self.index.query(
            vector=query_vec,
            top_k=k,
            include_metadata=True,
        )
        docs = []
        for match in results["matches"]:
            chunk_id  = match["id"]
            meta      = match.get("metadata", {})
            full_text = self.chunk_lookup.get(chunk_id, {}).get("text", "")
            if full_text:
                docs.append(Document(page_content=full_text, metadata={
                    **meta,
                    "score": round(match["score"], 4),
                }))
        return docs


# ---------------------------------------------------------------------------
# Hybrid retriever (dense + sparse BM25)
# ---------------------------------------------------------------------------

HYBRID_INDEX_NAME  = os.getenv("PINECONE_HYBRID_INDEX_NAME", "exorde-hybrid")
BM25_PARAMS_PATH   = os.path.join(os.path.dirname(__file__), "..", "bm25_params.json")


class HybridRetriever:
    """
    Combines dense (OpenAI) and sparse (BM25) retrieval via Pinecone hybrid search.

    Alpha controls the dense/sparse blend at query time:
        alpha=1.0  →  pure dense (equivalent to PineconeRetriever)
        alpha=0.5  →  equal blend
        alpha=0.0  →  pure sparse / keyword search

    The index must use metric='dotproduct' (created by ingest_hybrid.py).
    BM25 params are loaded from bm25_params.json (fitted by ingest_hybrid.py).
    """

    def __init__(self, alpha: float = 0.75):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.alpha = alpha
        self.embeddings = OpenAIEmbeddings(
            model=EMBED_MODEL,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = pc.Index(HYBRID_INDEX_NAME)
        self.chunk_lookup = {c["chunk_id"]: c for c in load_chunks()}

        # Load pre-fitted BM25 params
        from pinecone_text.sparse import BM25Encoder
        self.bm25 = BM25Encoder()
        self.bm25.load(BM25_PARAMS_PATH)

    def _scale(self, sparse_vec: dict) -> dict:
        """Scale sparse values by (1 - alpha)."""
        scale = 1.0 - self.alpha
        return {
            "indices": sparse_vec["indices"],
            "values":  [v * scale for v in sparse_vec["values"]],
        }

    def retrieve(self, query: str, k: int = 10) -> List[Document]:
        """
        Encode query with both dense and sparse encoders, blend with alpha,
        query Pinecone, and resolve full text from chunk_lookup.
        """
        # Dense vector scaled by alpha
        dense_raw  = self.embeddings.embed_query(query)
        dense_vec  = [v * self.alpha for v in dense_raw]

        # Sparse vector scaled by (1 - alpha)
        sparse_raw = self.bm25.encode_queries(query)
        sparse_vec = self._scale(sparse_raw)

        results = self.index.query(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=k,
            include_metadata=True,
        )

        docs = []
        for match in results["matches"]:
            chunk_id  = match["id"]
            meta      = match.get("metadata", {})
            full_text = self.chunk_lookup.get(chunk_id, {}).get("text", "")
            if full_text:
                docs.append(Document(page_content=full_text, metadata={
                    **meta,
                    "score": round(match["score"], 4),
                }))
        return docs
