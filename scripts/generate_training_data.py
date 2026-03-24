"""
Generates training data for fine-tuning SmolLM2.

Groups posts by theme, then uses GPT-4o-mini to write a question and answer for each
group. The format matches exactly what the model sees at inference time, so training
and production are aligned.

Output: training_data.jsonl — one JSON object per line with 'posts', 'question', 'answer'
Cost:   ~$0.50–$1.00 for 1000 examples at gpt-4o-mini pricing
"""

import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from config import CHAT_MODEL

load_dotenv()

logger = logging.getLogger(__name__)

# Config
random.seed(123)

N_EXAMPLES   = 1000   # number of training examples to generate
POSTS_PER_EX = 10     # posts per example — matches production k=10
OUTPUT_PATH  = Path("training_data.jsonl")
TEMPERATURE  = 0.8    # some variation in question style
RATE_LIMIT_S = 0.1    # sleep between API calls

# Prompts
SYSTEM_PROMPT = """\
You are generating training data for a social media analysis system.

Given a set of social media posts, generate:
1. A natural language question that someone might ask about the discussion in these posts
2. A concise 1-2 sentence answer grounded in the posts

Rules:
- The question must be answerable from the posts provided
- The answer must be grounded in the posts — do not add information not present
- The answer should synthesise across multiple posts, not just copy from one
- Write naturally, as if summarising a social media discussion
- Keep the answer to 1-2 sentences

Respond in JSON:
{"question": "...", "answer": "..."}
"""

# Data loading
def load_posts_by_theme() -> dict[str, list[str]]:
    """Load posts from the Exorde dataset and group them by theme."""
    sys.path.insert(0, str(Path(__file__).parent))
    from rag.preprocessing import load_chunks

    logger.info("Loading Exorde dataset...")
    chunks = load_chunks()

    by_theme: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        theme = chunk.get("metadata", {}).get("primary_theme", "")
        if theme:
            by_theme[theme].append(chunk["text"])

    # Keep only themes with enough posts to sample from
    by_theme = {t: posts for t, posts in by_theme.items() if len(posts) >= POSTS_PER_EX}
    logger.info("Found %d themes with ≥%d posts.", len(by_theme), POSTS_PER_EX)
    return by_theme

# Generation
def generate_example(posts: list[str], client: OpenAI) -> dict | None:
    """Ask GPT-4o-mini to write a question and answer for a set of posts."""
    posts_text = "\n".join(f"Post {i+1}: {p}" for i, p in enumerate(posts))
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Posts:\n{posts_text}"},
            ],
            response_format={"type": "json_object"},
            temperature=TEMPERATURE,
        )
        result = json.loads(response.choices[0].message.content)
        if "question" in result and "answer" in result:
            return {
                "posts":    posts,
                "question": result["question"].strip(),
                "answer":   result["answer"].strip(),
            }
    except Exception as e:
        logger.warning("Generation failed: %s", e)
    return None

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    by_theme = load_posts_by_theme()
    themes   = list(by_theme.keys())

    examples = []
    failed   = 0

    logger.info("Generating %d training examples...", N_EXAMPLES)
    for i in range(N_EXAMPLES):
        theme = random.choice(themes)
        posts = random.sample(by_theme[theme], POSTS_PER_EX)
        ex    = generate_example(posts, client)
        if ex:
            examples.append(ex)
        else:
            failed += 1

        if (i + 1) % 100 == 0:
            logger.info("%d/%d — %d saved, %d failed", i + 1, N_EXAMPLES, len(examples), failed)

        time.sleep(RATE_LIMIT_S)

    OUTPUT_PATH.write_text("\n".join(json.dumps(ex) for ex in examples))
    logger.info("Done. %d examples saved to %s", len(examples), OUTPUT_PATH)
    if failed:
        logger.warning("%d examples failed to generate.", failed)


if __name__ == "__main__":
    main()
