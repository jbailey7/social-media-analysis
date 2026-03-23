"""
Generates synthetic training data for fine-tuning SmolLM2-360M on the actual task.

Samples groups of thematically related posts from the Exorde dataset and uses
GPT-4o-mini to generate (question, answer) pairs grounded in those posts. The
resulting dataset directly matches the model's inference task: answer a natural
language question from retrieved social media posts.

Output:
    training_data.jsonl   One JSON object per line, each with 'posts', 'question', 'answer'

Usage:
    python generate_training_data.py

Cost estimate: ~1000 examples at gpt-4o-mini pricing ≈ $0.50–$1.00
"""

import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

random.seed(42)

N_EXAMPLES   = 1000   # number of training examples to generate
POSTS_PER_EX = 10     # posts per example — matches production k=10
OUTPUT_PATH  = Path("training_data.jsonl")
TEMPERATURE  = 0.8    # some variation in question style
RATE_LIMIT_S = 0.1    # sleep between API calls

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_posts_by_theme() -> dict[str, list[str]]:
    """Load Exorde posts and group by primary_theme."""
    sys.path.insert(0, str(Path(__file__).parent))
    from rag.preprocessing import load_and_clean_dataset

    print("Loading Exorde dataset...")
    chunks = load_and_clean_dataset(max_samples=50_000)

    by_theme: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        theme = chunk.get("primary_theme", "")
        if theme:
            by_theme[theme].append(chunk["text"])

    # Keep only themes with enough posts to sample from
    by_theme = {t: posts for t, posts in by_theme.items() if len(posts) >= POSTS_PER_EX}
    print(f"Found {len(by_theme)} themes with ≥{POSTS_PER_EX} posts.")
    return by_theme

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_example(posts: list[str], client: OpenAI) -> dict | None:
    """Call GPT-4o-mini to generate a (question, answer) pair for a set of posts."""
    posts_text = "\n".join(f"Post {i+1}: {p}" for i, p in enumerate(posts))
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
        print(f"  Warning: generation failed — {e}")
    return None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    by_theme = load_posts_by_theme()
    themes   = list(by_theme.keys())

    examples = []
    failed   = 0

    print(f"Generating {N_EXAMPLES} training examples...")
    for i in range(N_EXAMPLES):
        theme = random.choice(themes)
        posts = random.sample(by_theme[theme], POSTS_PER_EX)
        ex    = generate_example(posts, client)
        if ex:
            examples.append(ex)
        else:
            failed += 1

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{N_EXAMPLES} — {len(examples)} saved, {failed} failed")

        time.sleep(RATE_LIMIT_S)

    OUTPUT_PATH.write_text("\n".join(json.dumps(ex) for ex in examples))
    print(f"\nDone. {len(examples)} examples saved to {OUTPUT_PATH}")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
