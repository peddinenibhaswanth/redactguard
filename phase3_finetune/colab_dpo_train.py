# RedactGuard — Phase 3, Step B: DPO training on Google Colab
#
# RUN THIS AFTER colab_sft_train.py COMPLETES. DPO refines the SFT-adapted
# model's behavior on ambiguous/borderline cases — it does not replace SFT.
#
# HOW TO USE:
# 1. Same Colab notebook (or a new one) with T4 GPU still selected.
# 2. Make sure the SFT adapter folder from colab_sft_train.py is available at
#    SFT_ADAPTER_PATH below (same session: already there; new session: re-upload
#    or pull from Drive).
# 3. Upload data/dpo_pairs.jsonl (from Step 4.3 of the guide) to /content/data/.
#
# VERSION NOTE: see colab_sft_train.py's note - these pins are mid-2024, bump
# the specific package an install/import error names rather than assuming
# the whole set is broken.

# %%
# --- Cell 1: install deps (skip if already installed in this session) ---
!pip install -q -U transformers==4.44.0 peft==0.12.0 trl==0.9.6 \
    bitsandbytes==0.43.3 accelerate==0.33.0 datasets==2.20.0

# %%
# --- Cell 2: imports and config ---
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig
from trl import DPOTrainer, DPOConfig

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_ADAPTER_PATH = "/content/redactguard_sft_adapter"   # output of Step A
DPO_DATA_PATH = "/content/data/dpo_pairs.jsonl"
OUTPUT_DIR = "/content/redactguard_dpo_adapter"

assert torch.cuda.is_available(), \
    "No GPU detected — go to Runtime > Change runtime type > T4 GPU, then re-run."

# %%
# --- Cell 3: load base model + apply the SFT adapter as the starting point ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(SFT_ADAPTER_PATH)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# Load the SFT LoRA weights on top of the base model — this is the model DPO
# will refine. TRL's DPOTrainer will internally create a frozen reference copy.
model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, is_trainable=True)

# %%
# --- Cell 4: load DPO pairs dataset ---
# Expected format: dpo_pairs.jsonl, one JSON object per line:
#   {"prompt": "...", "chosen": "...", "rejected": "..."}
# (see Step 4.3 — chosen = flag as sensitive, rejected = let it through,
#  built specifically from ambiguous/indirect-reference cases)
dataset = load_dataset("json", data_files=DPO_DATA_PATH, split="train")
print(f"Loaded {len(dataset)} DPO pairs")
print("Example prompt:\n", dataset[0]["prompt"][:300])
print("Chosen:", dataset[0]["chosen"])
print("Rejected:", dataset[0]["rejected"])

# %%
# --- Cell 5: train ---
# Note: beta controls how strongly DPO pulls the model toward "chosen" —
# 0.1 is a reasonable default for a first pass; raise it (e.g. 0.3) if the
# post-DPO model doesn't seem to have shifted its behavior on eval.
dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=5e-5,
    logging_steps=5,
    save_strategy="epoch",
    bf16=True,
    beta=0.1,
    max_prompt_length=384,
    max_length=512,
    report_to="none",
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,  # TRL derives the frozen reference from the PEFT base automatically
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()

# %%
# --- Cell 6: save final adapter ---
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"DPO-refined adapter saved to {OUTPUT_DIR}")

# %%
# --- Cell 7: download the adapter folder to use locally ---
# Zip and download — this is small (tens of MB), not the full base model.
import shutil
shutil.make_archive("/content/redactguard_dpo_adapter", "zip", OUTPUT_DIR)

from google.colab import files
files.download("/content/redactguard_dpo_adapter.zip")

# %%
# --- Cell 8: before/after sanity check on an ambiguous case ---
test_input = tokenizer.apply_chat_template(
    [{"role": "system", "content": "You are a PII detection specialist. Given a text span from a document, classify whether it contains sensitive information."},
     {"role": "user", "content": 'Document context: "The property was transferred to the promoter\'s spouse last year."\nSpan to classify: "the promoter\'s spouse"'}],
    tokenize=False, add_generation_prompt=True,
)
inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))
# Expect: {"sensitive": true, ...} — DPO should have pushed the model toward
# flagging this indirect reference, which plain SFT alone often misses since
# it's not a clean-cut pattern.

# %%
# --- Cell 9: next step ---
print("""
Next steps (back on your laptop, in the main repo — see IMPLEMENTATION_GUIDE.md
Section 4.6):
1. Unzip the downloaded adapter into redactguard/phase3_finetune/final_adapter/
2. Implement app/detection/local_model_detector.py with the same interface as
   app/detection/llm_detector.py (detect_sensitive_spans(text, already_found)),
   loading this adapter with peft.PeftModel + the base Qwen2.5-0.5B model
   (CPU inference — no GPU needed for a model this size).
3. Swap it into the "llm_detect" node in app/graph/pipeline_graph.py.
4. Re-run eval/run_eval.py unchanged, save as eval/results_phase3.json.
5. Write the API-model vs local-model comparison table (precision/recall/F1,
   latency, offline capability) — this table is the actual Phase 3 deliverable.
""")
