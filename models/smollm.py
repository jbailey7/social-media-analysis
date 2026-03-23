"""
Loads the LoRA fine-tuned SmolLM2-360M-Instruct.

The model is fine-tuned on synthetic (posts, question, answer) triples generated
from the Exorde dataset, teaching it to answer natural language questions from
retrieved social media posts — the same task it performs in the production pipeline.

The fine-tuned LoRA adapter weights are included in this repository at:
  fine_tuned_model/
"""

import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"

# LoRA adapter weights are stored in this repository
FINE_TUNED_PATH = str(Path(__file__).resolve().parents[1] / "fine_tuned_model")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    """
    Load the base SmolLM2-360M and apply the LoRA fine-tuned adapter.
    Returns (model, tokenizer).
    """
    if not Path(FINE_TUNED_PATH).exists():
        raise FileNotFoundError(
            f"Fine-tuned model not found at: {FINE_TUNED_PATH}\n"
            "Ensure the fine_tuned_model/ directory is present in this repository."
        )

    print(f"Loading SmolLM2-360M base from {MODEL_ID}...")
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

    print(f"Applying LoRA adapter from {FINE_TUNED_PATH}...")
    model = PeftModel.from_pretrained(base_model, FINE_TUNED_PATH)
    model.eval()
    print("SmolLM2-360M (fine-tuned) ready.")
    return model, tokenizer


def load_base_model():
    """
    Load the base SmolLM2-360M-Instruct without any LoRA adapter.
    Used for evaluation config C (advanced agentic RAG without fine-tuning).
    Returns (model, tokenizer).
    """
    print(f"Loading SmolLM2-360M base (no LoRA) from {MODEL_ID}...")
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
    print("SmolLM2-360M (base, no fine-tuning) ready.")
    return model, tokenizer


def generate_summary(model, tokenizer, context: str, max_new_tokens: int = 150) -> str:
    """
    Generate an answer from the provided context.

    The context is pre-formatted by answer_node as:
      'Here are social media posts from December 2024:\\n\\nPost 1: ...\\n\\nQuestion: ...'

    This matches the format used during fine-tuning, ensuring consistent output.
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
    # apply_chat_template may return a dict or a tensor depending on version
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
