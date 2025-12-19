# Cell 2 — imports and environment detection
import os
import json
import torch
from pathlib import Path

has_gpu = torch.cuda.is_available()
device = torch.device("cuda" if has_gpu else "cpu")
print(f"GPU available: {has_gpu} — using device: {device}")

JSONL_PATH ="combined_cleaned.jsonl"


LARGE_MODEL_ID = "openai/gpt-oss-20b"
FALLBACK_MODEL_ID = "gpt2"

EPOCHS = 1
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
OUTPUT_DIR = "sft_output"


EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SIMILARITY_THRESHOLD = 0.65

# Cell 4 — load JSONL and normalize to fields 'input' and 'output'

from datasets import Dataset
from pathlib import Path
import json

# ---- safety check ----
if not Path(JSONL_PATH).exists():
    raise FileNotFoundError(f"JSONL file not found at {JSONL_PATH}. Upload it to Colab.")

# ---- normalization logic ----
def normalize_line(obj):
    """
    Expect HF chat format:
    {
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """
    if "messages" not in obj:
        raise ValueError("Missing 'messages' key")

    msgs = obj["messages"]

    if not isinstance(msgs, list) or len(msgs) < 2:
        raise ValueError("'messages' must be a list with >= 2 entries")

    user_msg = next(
        (m.get("content") for m in msgs if m.get("role") == "user"),
        None
    )
    assistant_msg = next(
        (m.get("content") for m in msgs if m.get("role") == "assistant"),
        None
    )

    if user_msg is None or assistant_msg is None:
        raise ValueError("Both user and assistant messages are required")

    return {
        "input": user_msg,
        "output": assistant_msg
    }

# ---- robust JSONL loader ----
records = []
skipped = 0

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line_no, raw_line in enumerate(f, start=1):
        line = raw_line.strip()

        # skip empty lines
        if not line:
            continue

        try:
            obj = json.loads(line)
            norm = normalize_line(obj)
            records.append(norm)

        except json.JSONDecodeError as e:
            skipped += 1
            print(f"[JSON ERROR] Line {line_no}: {e}")
            continue

        except Exception as e:
            skipped += 1
            print(f"[SKIPPED] Line {line_no}: {e}")
            continue

# ---- final validation ----
if not records:
    raise ValueError("No valid training records found.")

dataset = Dataset.from_list(records)

print(f"Loaded {len(dataset)} examples.")
print(f"Skipped {skipped} malformed records.")
print("Sample record:")
print(dataset[0])

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = LARGE_MODEL_ID if has_gpu else FALLBACK_MODEL_ID

tokenizer = AutoTokenizer.from_pretrained(model_id)

if has_gpu:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
else:
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)

print("Model loaded:", model_id)

# Cell 6 — prepare processing function compatible with SFTTrainer (tokenizer as 'processing_class' expects callables)
max_length = 2048

def preprocess_function(example):

    prompt = example["input"].strip()
    target = example["output"].strip()
    full = prompt + "\n\n### Response:\n" + target
    tokenized = tokenizer(full, truncation=True, max_length=max_length)
    return {"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"]}

tokenized_ds = dataset.map(preprocess_function, remove_columns=dataset.column_names)
print("Tokenized dataset example input_ids length:", len(tokenized_ds[0]["input_ids"]))

# Cell 7 — Auto-detect modules for LoRA
from peft import LoraConfig, get_peft_model
import re

def guess_lora_targets(model):
    """
    Automatically detect linear layers for common GPT architectures.
    """
    module_names = []
    for name, module in model.named_modules():
        if "attn" in name.lower() and isinstance(module, torch.nn.Linear):
            module_names.append(name)
        if "mlp" in name.lower() and isinstance(module, torch.nn.Linear):
            module_names.append(name)


    unique = list({n.split(".")[-1] for n in module_names})
    print("Detected LoRA target modules:", unique)
    return unique


target_modules = guess_lora_targets(model)

if len(target_modules) == 0:

    target_modules = ["c_attn", "c_proj"]

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=target_modules,
    bias="none",
    task_type="CAUSAL_LM",
)

peft_model = get_peft_model(model, peft_config)
print("PEFT model initialized.")
peft_model.print_trainable_parameters()

# Cell 8 — training using trl.SFTTrainer (auto precision detection)
import torch
from trl import SFTConfig, SFTTrainer
import os
os.environ["DISABLE_TRACKIO"] = "1"


has_gpu = torch.cuda.is_available()
bf16_supported = False
fp16_supported = False

if has_gpu:

    try:
        bf16_supported = torch.cuda.is_bf16_supported()
    except Exception:

        try:
            props = torch.cuda.get_device_properties(0)

            bf16_supported = props.major >= 8
        except Exception:
            bf16_supported = False

    fp16_supported = True

use_bf16 = bool(bf16_supported)
use_fp16 = (not use_bf16) and bool(fp16_supported)
use_cpu = not has_gpu

# Adjust batch size for CPU
train_batch = BATCH_SIZE
if use_cpu:
    train_batch = max(1, BATCH_SIZE // 8)
    print(f"Running on CPU: reducing per-device batch size to {train_batch}")

print(f"Device: {'GPU' if has_gpu else 'CPU'}, bf16_supported={bf16_supported}, fp16_supported={fp16_supported}")
print(f"Training precision chosen: bf16={use_bf16}, fp16={use_fp16}, use_cpu={use_cpu}")

training_args = SFTConfig(
    learning_rate=LEARNING_RATE,
    gradient_checkpointing=True,
    num_train_epochs=EPOCHS,
    logging_steps=10,
    per_device_train_batch_size=train_batch,
    gradient_accumulation_steps=1,
    max_length=max_length,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    output_dir=OUTPUT_DIR,
    report_to=[],        # <-- Disable Trackio, WandB, etc.
    push_to_hub=False,
    bf16=use_bf16,
    fp16=use_fp16,
    use_cpu=use_cpu,
)



trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_ds,
    processing_class=tokenizer,
)


trainer.train()
trainer.save_model(OUTPUT_DIR)
print("Training complete. Model saved to", OUTPUT_DIR)

# Cell 10 — embeddings creation and constrained inference wrapper
from sentence_transformers import SentenceTransformer
import numpy as np
import torch.nn.functional as F


embedder = SentenceTransformer(EMB_MODEL)

dataset_texts = [rec["input"] for rec in records]
dataset_outputs = [rec["output"] for rec in records]


dataset_embeddings = embedder.encode(dataset_texts, convert_to_numpy=True, show_progress_bar=True)

dataset_embeddings = dataset_embeddings / np.linalg.norm(dataset_embeddings, axis=1, keepdims=True)

def find_best_match(query, top_k=1):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    sims = np.dot(dataset_embeddings, q_emb[0])
    idx = int(np.argmax(sims))
    score = float(np.max(sims))
    return idx, score

def generate_answer_from_dataset(query, similarity_threshold=SIMILARITY_THRESHOLD):
    idx, score = find_best_match(query)
    if score < similarity_threshold:
        return "sorry"

    return dataset_outputs[idx]

# Cell 11 — test inference
user_questions = ["What makes Unitic’s INR deposit and withdrawal system stand out?"]

for q in user_questions:
    answer = generate_answer_from_dataset(q)
    print("Q:", q)
    print("A:", answer)
    print("----")
