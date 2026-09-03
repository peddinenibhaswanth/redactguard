# RedactGuard — Phase 3, Step A: SFT training on Google Colab
#
# HOW TO USE:
# 1. Open a new Colab notebook (colab.research.google.com)
# 2. Runtime -> Change runtime type -> T4 GPU  (do this BEFORE running any cell)
# 3. Split this file at the "# %%" markers and paste each block into its own
#    Colab cell, in order.
#    Do NOT upload this file and run `!python colab_sft_train.py` - the
#    `!pip install` line below is Jupyter cell magic, not valid Python, so
#    running it as a plain script fails immediately.
# 4. Upload data/sft_train.jsonl to /content/data/ before the training cell -
#    drag-and-drop into the Colab file browser sidebar (folder icon on the
#    left), or mount Google Drive and copy it in.
#
# Versions are intentionally NOT pinned. This script targets the current TRL
# API (>=1.0): trainers take `processing_class` rather than `tokenizer`, and
# SFTConfig takes `max_length` rather than `max_seq_length`. Old pinned
# versions (trl 0.9.x) use the opposite names and will TypeError here.

# %%
# --- Cell 1: install deps ---
!pip install -q -U transformers peft trl bitsandbytes accelerate datasets

# %%
# --- Cell 2: imports and config ---
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = "/content/data/sft_train.jsonl"       # from prepare_sft_dataset.py
OUTPUT_DIR = "/content/redactguard_sft_adapter"

assert torch.cuda.is_available(), \
    "No GPU detected — go to Runtime > Change runtime type > T4 GPU, then re-run."

# A T4 is Turing (compute capability 7.5) and has NO bfloat16 support - that
# needs Ampere (8.0+). Forcing bf16=True on a T4 errors out or silently
# degrades, so pick the dtype from what the GPU actually supports.
BF16_OK = torch.cuda.is_bf16_supported()
COMPUTE_DTYPE = torch.bfloat16 if BF16_OK else torch.float16

print("GPU:", torch.cuda.get_device_name(0))
print("bf16 supported:", BF16_OK, "-> training in", "bf16" if BF16_OK else "fp16")

# %%
# --- Cell 3: load base model in 4-bit (QLoRA) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
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
# --- Cell 4: prepare for k-bit training, then attach LoRA adapters ---
# prepare_model_for_kbit_training is NOT optional for QLoRA. It casts the
# layernorms and upcasts the LM head to fp32, which is exactly what keeps
# fp16 training numerically stable on a 4-bit base. Skipping it and training
# in fp16 (which a T4 forces, since it has no bf16) makes the model diverge:
# loss blows up and generation collapses into a single repeated token.
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen2.5 attention projections
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Sanity check: trainable params should be well under 1% of the total.
# If it prints ~100%, target_modules doesn't match this architecture -
# inspect model.named_modules() to find the real projection names.

# %%
# --- Cell 5: load dataset ---
# Expected format: one JSON object per line, {"text": "<formatted ChatML example>"}
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
    # 1e-4 rather than the 2e-4 often quoted for QLoRA: that figure assumes
    # bf16, and fp16 (forced by the T4) has a much narrower dynamic range.
    # Combined with warmup below this trains just as well and stops the loss
    # from spiking in the first few steps.
    learning_rate=1e-4,
    # ~150 total steps here (808 examples / effective batch 16, 3 epochs), so
    # 10 warmup steps is a short ramp. This version of SFTConfig takes
    # warmup_steps; warmup_ratio does not exist on it.
    warmup_steps=10,
    max_grad_norm=0.3,
    optim="paged_adamw_8bit",   # standard QLoRA optimizer; also eases T4 memory
    logging_steps=5,            # frequent enough to actually see a loss curve
    save_strategy="epoch",
    bf16=BF16_OK,
    fp16=not BF16_OK,
    dataset_text_field="text",
    max_length=512,           # named max_seq_length in trl < 1.0
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer,   # named `tokenizer` in trl < 1.0
)

trainer.train()

# Read the loss column printed above before continuing. It should fall from
# roughly 1-2 down toward ~0.1-0.4. If it goes to nan, or sits flat, or jumps
# by orders of magnitude, training diverged - do not proceed to DPO on a
# diverged checkpoint. Halve the learning rate and re-run this cell.
final_loss = trainer.state.log_history[-1].get("train_loss")
print("final train_loss:", final_loss)
assert final_loss is not None and final_loss == final_loss, \
    "train_loss is nan - training diverged. Lower learning_rate and re-run."

# %%
# --- Cell 7: save adapter (small — tens of MB, not the full model) ---
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"SFT adapter saved to {OUTPUT_DIR}")
print("Next: run colab_dpo_train.py in this same session, with "
      "SFT_ADAPTER_PATH pointing at this folder.")

# %%
# --- Cell 8: quick sanity check before moving to DPO ---
test_input = tokenizer.apply_chat_template(
    [{"role": "system", "content": "You are a PII detection specialist. Given a text span from a document, classify whether it contains sensitive information."},
     {"role": "user", "content": 'Document context: "Contact the director at ramesh.kumar@example.com"\nSpan to classify: "ramesh.kumar@example.com"'}],
    tokenize=False, add_generation_prompt=True,
)
inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))
# Expect roughly: {"sensitive": true, "category": "email", "confidence": ...}
# Gibberish or ignored JSON format means too few examples or too few epochs.
