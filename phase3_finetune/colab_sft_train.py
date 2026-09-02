# RedactGuard — Phase 3, Step A: SFT training on Google Colab
#
# HOW TO USE:
# 1. Open a new Colab notebook (colab.research.google.com)
# 2. Runtime -> Change runtime type -> T4 GPU  (do this BEFORE running any cell)
# 3. Split this file at the "# %%" markers, paste each block into its own cell,
#    in order. Or upload this whole file and run: !python colab_sft_train.py
# 4. Upload data/sft_train.jsonl (from Step 4.2 of the guide) to /content/data/
#    before running the training cell — either drag-and-drop in the Colab file
#    browser sidebar, or mount Google Drive and copy it in.
#
# VERSION NOTE: the pins below (transformers 4.44.0, trl 0.9.6, etc.) are
# mid-2024 releases. If Cell 1 fails to install, or Cell 3/4 throws an
# unfamiliar error, it's likely Colab's preinstalled CUDA/torch has moved on
# since these were pinned — check the error message, bump the specific
# package it names, and re-run from that cell. Don't assume the pins are
# broken across the board just because one is.

# %%
# --- Cell 1: install deps (Colab has some of these; pin versions to avoid drift) ---
!pip install -q -U transformers==4.44.0 peft==0.12.0 trl==0.9.6 \
    bitsandbytes==0.43.3 accelerate==0.33.0 datasets==2.20.0

# %%
# --- Cell 2: imports and config ---
import torch
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = "/content/data/sft_train.jsonl"       # from Step 4.2
OUTPUT_DIR = "/content/redactguard_sft_adapter"

assert torch.cuda.is_available(), \
    "No GPU detected — go to Runtime > Change runtime type > T4 GPU, then re-run."

# %%
# --- Cell 3: load base model in 4-bit (QLoRA) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# %%
# --- Cell 4: attach LoRA adapters ---
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen2.5 attention proj names
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Sanity check: this should print well under 1% of total params as trainable.
# If it prints 100%, target_modules names are wrong for the loaded model's
# architecture — check model.named_modules() to find the correct proj names.

# %%
# --- Cell 5: load dataset ---
# Expected format: sft_train.jsonl, one JSON object per line:
#   {"text": "<full formatted SFT_TEMPLATE string from Step 4.2>"}
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
print(f"Loaded {len(dataset)} SFT examples")
print("Example:\n", dataset[0]["text"][:500])

# %%
# --- Cell 6: train ---
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    dataset_text_field="text",
    max_seq_length=512,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()

# %%
# --- Cell 7: save adapter (small — tens of MB, not the full model) ---
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"SFT adapter saved to {OUTPUT_DIR}")
print("Next: zip this folder and download it, or copy to Drive, then run "
      "colab_dpo_train.py pointing SFT_ADAPTER_PATH at this folder.")

# %%
# --- Cell 8: quick manual sanity check before moving to DPO ---
from peft import PeftModel
test_input = tokenizer.apply_chat_template(
    [{"role": "system", "content": "You are a PII detection specialist. Given a text span from a document, classify whether it contains sensitive information."},
     {"role": "user", "content": 'Document context: "Contact the director at ramesh.kumar@example.com"\nSpan to classify: "ramesh.kumar@example.com"'}],
    tokenize=False, add_generation_prompt=True,
)
inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))
# Expect roughly: {"sensitive": true, "category": "email", "confidence": ...}
# If output is gibberish or ignores the JSON format, increase epochs or check
# that sft_train.jsonl actually has enough examples (Step 4.2 floor: 300-600).
