"""
Fine-tunes SmolLM2-360M-Instruct on the SAMSum dialogue summarization dataset
using LoRA (PEFT). Saves the adapter weights to fine_tuned_model/.

The fine-tuned model is used by the answer_node in the agentic pipeline to
summarize retrieved social media posts into a final answer.

Usage:
    python train.py

Requirements:
    - Python 3.11 or 3.12
    - GPU recommended (~4 minutes on CUDA, much longer on CPU)
    - No API keys required

Output:
    fine_tuned_model/   LoRA adapter weights + tokenizer (overwrites existing)
    checkpoints/        Per-epoch Trainer checkpoints (can be deleted after training)
"""

import random

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

random.seed(42)
torch.manual_seed(42)

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID  = "HuggingFaceTB/SmolLM2-360M-Instruct"
SAVE_PATH = "./fine_tuned_model"

NUM_TRAIN  = 1000
MAX_LENGTH = 512

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

print("Loading SAMSum dataset...")
dataset      = load_dataset("knkarthick/samsum")
train_samples = dataset["train"].shuffle(seed=42).select(range(NUM_TRAIN))

# ---------------------------------------------------------------------------
# Model and tokenizer
# ---------------------------------------------------------------------------

print(f"Loading base model from {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
)
if DEVICE == "cpu":
    base_model = base_model.to(DEVICE)

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def _to_ids(result):
    """apply_chat_template returns a list or BatchEncoding depending on version."""
    return result if isinstance(result, list) else result["input_ids"]


def tokenize(example):
    """
    Encode a (dialogue, summary) pair as a chat sequence.
    Labels are masked on the prompt tokens so loss is computed only on the
    summary (assistant) portion.
    """
    messages = [{
        "role": "user",
        "content": (
            "Summarize the following conversation in one or two sentences.\n\n"
            f"Conversation:\n{example['dialogue']}"
        ),
    }]

    # Prompt-only length (used to mask labels)
    prompt_ids = _to_ids(tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    ))
    prompt_len = len(prompt_ids)

    # Full sequence: prompt + summary
    full_ids = _to_ids(tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": example["summary"]}],
        add_generation_prompt=False,
        tokenize=True,
    ))

    if len(full_ids) > MAX_LENGTH:
        full_ids = full_ids[:MAX_LENGTH]

    pad_len        = MAX_LENGTH - len(full_ids)
    input_ids      = full_ids + [tokenizer.pad_token_id] * pad_len
    attention_mask = [1] * len(full_ids) + [0] * pad_len

    # Mask prompt tokens and padding from the loss
    labels = full_ids.copy() + [-100] * pad_len
    for j in range(min(prompt_len, len(full_ids))):
        labels[j] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


print("Tokenizing training data...")
tokenized_train = train_samples.map(tokenize, remove_columns=train_samples.column_names)
tokenized_train.set_format("torch")

# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=(DEVICE == "cuda"),
    logging_steps=25,
    save_strategy="epoch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
)

print(f"\nTraining on {DEVICE}...")
trainer.train()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

print(f"\nSaving adapter weights to {SAVE_PATH}...")
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print("Done. Run the app with: streamlit run app.py")
