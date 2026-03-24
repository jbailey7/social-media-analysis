"""
Shared preprocessing for the Exorde dataset.

Centralises dataset loading and text cleaning so that ingestion and retrieval
use identical preprocessing. chunk_ids encode the raw dataset row index,
making them stable across both steps.
"""

import hashlib
import logging
import re

from datasets import load_dataset

logger = logging.getLogger(__name__)

DATASET_NAME = "Exorde/exorde-social-media-december-2024-week1"
N_SAMPLES    = 50_000
MIN_TEXT_LEN = 20

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def clean_post(text: str) -> str:
    """Remove URLs and normalise whitespace."""
    if text is None:
        return ""
    text = _URL_RE.sub("", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_chunks(verbose: bool = True) -> list:
    """
    Load and preprocess the Exorde dataset into a list of chunk dicts.

    Each chunk has the shape:
        {
            "chunk_id": "exorde_<i>",
            "text":     <cleaned post text>,
            "metadata": {lang, timestamp, primary_theme, sentiment},
        }

    The chunk_id encodes the raw dataset row index, making it stable across
    ingestion and retrieval.
    """
    if verbose:
        logger.info("Loading Exorde dataset from HuggingFace...")

    dataset = load_dataset(DATASET_NAME, split="train")

    chunks     = []
    seen_texts = set()

    for i in range(min(N_SAMPLES * 3, len(dataset))):
        if len(chunks) >= N_SAMPLES:
            break

        row = dataset[i]
        if row.get("language") != "en":
            continue

        text = row.get("original_text", "")
        if not isinstance(text, str) or len(text.strip()) < MIN_TEXT_LEN:
            continue

        text = clean_post(text)
        if len(text) < MIN_TEXT_LEN:
            continue

        normalized = " ".join(text.lower().split())
        text_hash  = hashlib.md5(normalized.encode()).hexdigest()
        if text_hash in seen_texts:
            continue
        seen_texts.add(text_hash)

        chunks.append({
            "chunk_id": f"exorde_{i}",
            "text": text,
            "metadata": {
                "lang":          row.get("language", ""),
                "timestamp":     row.get("date", ""),
                "primary_theme": row.get("primary_theme", ""),
                "sentiment":     row.get("sentiment", 0.0),
            },
        })

    if verbose:
        logger.info("Loaded %s chunks.", f"{len(chunks):,}")

    return chunks
