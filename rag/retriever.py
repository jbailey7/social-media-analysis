"""
Pinecone retriever.

The Pinecone index stores only lightweight metadata (no full text).
Full text is resolved via chunk_lookup, a dict rebuilt from the Exorde
dataset using the same preprocessing as ingest.py so chunk IDs match.
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
