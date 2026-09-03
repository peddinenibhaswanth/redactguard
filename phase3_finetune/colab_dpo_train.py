# RedactGuard — Phase 3, Step B: DPO training on Google Colab
#
# RUN THIS AFTER colab_sft_train.py COMPLETES. DPO refines the SFT-adapted
# model's behaviour on ambiguous/borderline cases — it does not replace SFT,
# and starting DPO from the base model instead of the SFT checkpoint produces
# a poorly behaved model.
#
# HOW TO USE:
# 1. Same Colab notebook as Step A, T4 GPU still selected. Staying in the same
#    session matters: SFT_ADAPTER_PATH below points at /content, which is wiped
#    when the runtime disconnects. New session -> re-upload the adapter folder
#    (or pull it from Drive) and adjust the path.
# 2. Paste the cells below, in order, after Step A's cells.
# 3. Upload data/dpo_pairs.jsonl to /content/data/.
#
# Targets the current TRL API (>=1.0), same as Step A: `processing_class`
# rather than `tokenizer`, and DPOConfig no longer accepts `max_prompt_length`.

# %%
# --- Cell 1: install deps (skip if Step A already ran in this session) ---
!pip install -q -U transformers peft trl bitsandbytes accelerate datasets

# %%
# --- Cell 2: imports and config ---
import torch
from datasets import load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_ADAPTER_PATH = "/content/redactguard_sft_adapter"   # output of Step A
DPO_DATA_PATH = "/content/data/dpo_pairs.jsonl"
OUTPUT_DIR = "/content/redactguard_dpo_adapter"

assert torch.cuda.is_available(), \
    "No GPU detected — go to Runtime > Change runtime type > T4 GPU, then re-run."

# T4 (Turing) has no bfloat16 support; that needs Ampere or newer.
BF16_OK = torch.cuda.is_bf16_supported()
COMPUTE_DTYPE = torch.bfloat16 if BF16_OK else torch.float16

print("GPU:", torch.cuda.get_device_name(0))
print("bf16 supported:", BF16_OK, "-> training in", "bf16" if BF16_OK else "fp16")

# %%
# --- Cell 3: load base model + apply the SFT adapter as the starting point ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
)

tokenizer = AutoTokenizer.from_pretrained(SFT_ADAPTER_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# Same fp32 layernorm/LM-head casting as Step A - required for stable fp16
# training on a 4-bit base, and just as necessary here as it is for SFT.
base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

# Load the SFT LoRA weights on top of the base model - this is what DPO
# refines. is_trainable=True is required, otherwise the adapter loads frozen
# and training silently updates nothing.
model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, is_trainable=True)

# %%
# --- Cell 4: load DPO pairs dataset ---
# Expected format: one JSON object per line with prompt / chosen / rejected.
# Built from ambiguous indirect-reference cases: chosen = flag as sensitive,
# rejected = let it through. DPO is for borderline calls, not obvious ones.
dataset = load_dataset("json", data_files=DPO_DATA_PATH, split="train")
print(f"Loaded {len(dataset)} DPO pairs")
print("Example prompt:\n", dataset[0]["prompt"][:300])
print("Chosen:  ", dataset[0]["chosen"])
print("Rejected:", dataset[0]["rejected"])

# %%
# --- Cell 5: train ---
# beta controls how hard DPO pulls toward "chosen". 0.1 is a sane first pass;
# raise it (e.g. 0.3) if post-DPO behaviour on eval looks unchanged.
dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=5e-5,
    warmup_ratio=0.03,
    max_grad_norm=0.3,
    optim="paged_adamw_8bit",
    logging_steps=5,
    save_strategy="epoch",
    bf16=BF16_OK,
    fp16=not BF16_OK,
    beta=0.1,
    max_length=512,        # DPOConfig no longer takes max_prompt_length
    report_to="none",
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,   # with a PEFT model TRL derives the reference by disabling adapters
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,   # named `tokenizer` in trl < 1.0
)

trainer.train()

# %%
# --- Cell 6: save final adapter ---
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"DPO-refined adapter saved to {OUTPUT_DIR}")

# %%
# --- Cell 7: download the adapter to use locally ---
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
# Expect: {"sensitive": true, ...} — DPO should push toward flagging this
# indirect reference, which plain SFT often misses since it is not a clean
# pattern match.

# %%
# --- Cell 9: next steps ---
print("""
Back on your laptop, in the main repo (see README.md, Phase 3 section):
1. python -m phase3_finetune.export_adapter <path-to-downloaded-zip>
2. In app/graph/pipeline_graph.py swap the detector import to:
     from app.detection.local_model_detector import detect_local_model_spans as detect_llm_spans
3. pip install -r requirements-phase3.txt   (torch/transformers/peft for CPU inference)
4. Re-run eval/run_eval.py, save the output as eval/results_phase3.json
5. Write the API-model vs local-model comparison table - that table is the
   actual Phase 3 deliverable, not the training run itself.
""")
