"""Phase 3's drop-in replacement for llm_detector.py - same interface
(detect_sensitive_spans(text, already_found) -> List[dict]), loaded from the
LoRA adapter produced by phase3_finetune/colab_sft_train.py +
colab_dpo_train.py and placed at phase3_finetune/final_adapter/ via
export_adapter.py. CPU inference only - a model this size doesn't need a
GPU to run, only to train, and CPU inference is the actual point:
sensitive documents never leave the machine once this node is swapped in.

Honest limitation vs. llm_detector.py: the API-model prompt performs
open-ended extraction over free text ("find all sensitive spans in this
passage"). The fine-tuned model was trained as a classifier on a
(context, span) pair, not an extractor - it has no way to propose spans on
its own. So this module still needs *something* to propose candidate spans
before classifying them; it uses a cheap capitalized-phrase heuristic for
that, which will under-propose compared to the LLM's free-form extraction.
This tradeoff (and its effect on recall) is exactly what the Phase 3
results_phase3.json comparison table is supposed to surface honestly, not
paper over.
"""
import json
import os
import re
from functools import lru_cache
from typing import List

from app.config import CONFIDENCE_THRESHOLD, LOCAL_ADAPTER_PATH
from app.detection.base import FlaggedSpan, build_page_text_index, map_results_to_flagged
from app.detection.candidates import iter_candidates
from app.extraction.base import ExtractedDocument, TextSpan
from app.llm_client import strip_fences
from phase3_finetune.prompt_template import SYSTEM_MSG, context_window

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = LOCAL_ADAPTER_PATH
CANDIDATE_CONTEXT_WINDOW = 80

# The fine-tuned model occasionally emits a category outside the label set it
# was trained on (an observed DPO checkpoint produced "indication" instead of
# "indirect_reference"). Anything unrecognised is mapped to "other" rather
# than propagated, so the report's category counts stay meaningful - the
# sensitive/not-sensitive verdict is unaffected either way.
VALID_CATEGORIES = {
    "name", "address", "indirect_reference", "email", "pan", "aadhaar",
    "ifsc", "phone", "date_of_birth", "none", "other",
}

@lru_cache(maxsize=1)
def _load_model():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not os.path.isdir(ADAPTER_PATH):
        raise FileNotFoundError(
            f"No fine-tuned adapter found at {ADAPTER_PATH}. Run Phase 3 training on Colab "
            f"(phase3_finetune/colab_sft_train.py, then colab_dpo_train.py), then place the "
            f"result here with phase3_finetune/export_adapter.py."
        )

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.float32, device_map="cpu")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer


def _classify_span(context: str, span_text: str) -> dict:
    import torch

    model, tokenizer = _load_model()
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": f'Document context: "{context}"\nSpan to classify: "{span_text}"'},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
            pad_token_id=tokenizer.pad_token_id,
        )
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    # Extract the first {...} block rather than parsing the whole string. The
    # fine-tuned model reliably emits the right JSON but sometimes appends a
    # stray character after it (a trailing "." has been observed), and
    # json.loads on the raw string rejects that as "Extra data" - which would
    # silently downgrade every span to not-sensitive via the fallback below.
    cleaned = strip_fences(decoded.strip())
    match = re.search(r"\{.*?\}", cleaned, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"sensitive": False, "category": "other", "confidence": 0.0}


def detect_sensitive_spans(text: str, already_found: List[TextSpan]) -> List[dict]:
    """Same interface/contract as llm_detector.detect_sensitive_spans."""
    already_texts = {s.text.lower() for s in already_found}
    results = []

    # Same generator prepare_sft_dataset.py draws its negatives from, so the
    # model is asked about the distribution it was trained to reject.
    for start, end, candidate in iter_candidates(text):
        if candidate.lower() in already_texts:
            continue

        context = context_window(text, start, end, CANDIDATE_CONTEXT_WINDOW)
        classification = _classify_span(context, candidate)
        if classification.get("sensitive"):
            category = str(classification.get("category", "other"))
            results.append(
                {
                    "text": candidate,
                    "category": category if category in VALID_CATEGORIES else "other",
                    "confidence": float(classification.get("confidence", 0.5)),
                }
            )

    return results


def detect_local_model_spans(doc: ExtractedDocument, already_found: List[FlaggedSpan]) -> List[FlaggedSpan]:
    """Mirrors llm_detector.detect_llm_spans's structure exactly, so swapping
    this in for the llm_detect node in graph/pipeline_graph.py is a one-line
    change: `from app.detection.local_model_detector import
    detect_local_model_spans as detect_llm_spans`."""
    flagged: List[FlaggedSpan] = []
    page_nums = sorted({s.page_num for s in doc.spans})

    for page_num in page_nums:
        page_text, ranges = build_page_text_index(doc.spans, page_num)
        page_already_found = [s for s in already_found if s.page_num == page_num]
        results = detect_sensitive_spans(page_text, page_already_found)

        flagged.extend(
            map_results_to_flagged(
                results, page_text, ranges, page_num, "local_model", CONFIDENCE_THRESHOLD
            )
        )

    return flagged
