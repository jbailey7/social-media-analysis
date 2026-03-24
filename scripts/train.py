"""
Fine-tunes SmolLM2-360M-Instruct on the synthetic Q&A data using LoRA.

Run generate_training_data.py first to produce training_data.jsonl, then run this.
A GPU is recommended — about 4 minutes on CUDA, much longer on CPU. No API keys needed.

Output:
    fine_tuned_model/   LoRA adapter weights + tokenizer
    checkpoints/        Per-epoch checkpoints (can be deleted after training)
"""

import json
import logging
import random
from pathlib import Path

from config import SMOLLM_MODEL_ID

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

random.seed(42)
torch.manual_seed(42)

DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID         = SMOLLM_MODEL_ID
SAVE_PATH        = "./fine_tuned_model"
TRAINING_DATA    = Path("training_data.jsonl")
MAX_LENGTH       = 512

# Dataset
if not TRAINING_DATA.exists():
    raise FileNotFoundError(
        f"{TRAINING_DATA} not found. Run generate_training_data.py first."
    )

logger.info("Loading training data from %s...", TRAINING_DATA)
raw = [json.loads(line) for line in TRAINING_DATA.read_text().splitlines() if line.strip()]
dataset = Dataset.from_list(raw)
logger.info("Loaded %d training examples.", len(dataset))

# Model and tokenizer
logger.info("Loading base model from %s...", MODEL_ID)
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

# Tokenization
def _to_ids(result):
    """apply_chat_template returns either a list or a BatchEncoding — normalise to a list."""
    return result if isinstance(result, list) else result["input_ids"]


def tokenize(example):
    """
    Turn a training example into token IDs the Trainer can use.

    The prompt format matches what answer_node sends at inference time, so the
    model sees the same structure during training and production.
    Loss is only computed on the answer tokens — the prompt is masked out.
    """
    posts_text = "\n".join(
        f"Post {i+1}: {p}" for i, p in enumerate(example["posts"])
    )
    user_content = (
        "Here are social media posts from December 2024:\n\n"
        f"{posts_text}\n\n"
        f"Question: {example['question']}"
    )

    messages = [{"role": "user", "content": user_content}]

    # Tokenize the prompt alone so we know where it ends
    prompt_ids = _to_ids(tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    ))
    prompt_len = len(prompt_ids)

    # Tokenize the full prompt + answer together
    full_ids = _to_ids(tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": example["answer"]}],
        add_generation_prompt=False,
        tokenize=True,
    ))

    if len(full_ids) > MAX_LENGTH:
        full_ids = full_ids[:MAX_LENGTH]

    pad_len        = MAX_LENGTH - len(full_ids)
    input_ids      = full_ids + [tokenizer.pad_token_id] * pad_len
    attention_mask = [1] * len(full_ids) + [0] * pad_len

    # Set prompt tokens and padding to -100 so the loss ignores them
    labels = full_ids.copy() + [-100] * pad_len
    for j in range(min(prompt_len, len(full_ids))):
        labels[j] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


logger.info("Tokenizing training data...")
tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
tokenized.set_format("torch")

# LoRA
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

# Training
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
    train_dataset=tokenized,
)

logger.info("Training on %s...", DEVICE)
trainer.train()

# Save
logger.info("Saving adapter weights to %s...", SAVE_PATH)
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
logger.info("Done. Run the app with: streamlit run app.py")
