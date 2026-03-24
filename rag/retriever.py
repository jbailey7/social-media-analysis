"""
Two retriever classes:
  PineconeRetriever — dense vector search only (what the app uses)
  HybridRetriever   — dense + BM25 sparse, used in the hybrid search experiment

Both store full post text locally and look it up by chunk ID after the
Pinecone query, so Pinecone only needs to store lightweight metadata.
"""

import os
from typing import List

from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from rag.preprocessing import load_chunks
from config import EMBED_MODEL

load_dotenv()
INDEX_NAME  = os.getenv("PINECONE_INDEX_NAME", "exorde-week1")


class PineconeRetriever:
    """Dense vector retriever backed by Pinecone."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=EMBED_MODEL,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index        = pc.Index(INDEX_NAME)
        self.chunk_lookup = {c["chunk_id"]: c for c in load_chunks()}

    def retrieve(self, query: str, k: int = 10) -> List[Document]:
        """Embed the query, search Pinecone, and return the top-k posts as Documents."""
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


# Hybrid retriever (dense + sparse BM25)
HYBRID_INDEX_NAME  = os.getenv("PINECONE_HYBRID_INDEX_NAME", "exorde-hybrid")
BM25_PARAMS_PATH   = os.path.join(os.path.dirname(__file__), "..", "bm25_params.json")


class HybridRetriever:
    """
    Combines dense and BM25 sparse retrieval. Alpha controls the mix:
      1.0 = pure dense, 0.5 = equal blend, 0.0 = pure keyword search.

    Requires a dotproduct Pinecone index (created by ingest_hybrid.py) and
    pre-fitted BM25 params saved in bm25_params.json.
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
        """Multiply sparse values by (1 - alpha) to apply the blend weight."""
        scale = 1.0 - self.alpha
        return {
            "indices": sparse_vec["indices"],
            "values":  [v * scale for v in sparse_vec["values"]],
        }

    def retrieve(self, query: str, k: int = 10) -> List[Document]:
        """Encode with both dense and sparse, blend by alpha, and return top-k posts."""
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
