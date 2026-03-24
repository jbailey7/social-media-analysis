"""
Loads SmolLM2-360M-Instruct, with or without the LoRA adapter.

The fine-tuned version was trained on synthetic Q&A examples generated from the
same Exorde dataset the retrieval index uses, so the training task matches inference.
Adapter weights are stored in fine_tuned_model/.
"""

import logging
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import SMOLLM_MODEL_ID

logger = logging.getLogger(__name__)

MODEL_ID = SMOLLM_MODEL_ID

# LoRA adapter weights are stored in this repository
FINE_TUNED_PATH = str(Path(__file__).resolve().parents[1] / "fine_tuned_model")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    """Load the base model and apply the LoRA adapter. Returns (model, tokenizer)."""
    if not Path(FINE_TUNED_PATH).exists():
        raise FileNotFoundError(
            f"Fine-tuned model not found at: {FINE_TUNED_PATH}\n"
            "Ensure the fine_tuned_model/ directory is present in this repository."
        )

    logger.info("Loading SmolLM2-360M base from %s...", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    if DEVICE == "cpu":
        base_model = base_model.to(DEVICE)

    logger.info("Applying LoRA adapter from %s...", FINE_TUNED_PATH)
    model = PeftModel.from_pretrained(base_model, FINE_TUNED_PATH)
    model.eval()
    logger.info("SmolLM2-360M (fine-tuned) ready.")
    return model, tokenizer


def load_base_model():
    """Load the base model with no LoRA adapter. Used for eval config C. Returns (model, tokenizer)."""
    logger.info("Loading SmolLM2-360M base (no LoRA) from %s...", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    if DEVICE == "cpu":
        model = model.to(DEVICE)

    model.eval()
    logger.info("SmolLM2-360M (base, no fine-tuning) ready.")
    return model, tokenizer


def generate_summary(model, tokenizer, context: str, max_new_tokens: int = 150) -> str:
    """
    Generate an answer given the formatted context string.

    The context is built by answer_node in the same format used during fine-tuning,
    so the model knows what to expect.
    """
    messages = [
        {
            "role": "user",
            "content": context,
        }
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # apply_chat_template returns either a dict or a tensor depending on the transformers version
    if isinstance(inputs, dict):
        input_ids = inputs["input_ids"].to(model.device)
    else:
        input_ids = inputs.to(model.device)

    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
