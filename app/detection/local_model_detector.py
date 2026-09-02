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

from app.config import CONFIDENCE_THRESHOLD
from app.detection.base import FlaggedSpan, build_page_text_index, merge_bbox
from app.extraction.base import ExtractedDocument, TextSpan
from app.llm_client import strip_fences
from phase3_finetune.prompt_template import SYSTEM_MSG, context_window

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "phase3_finetune", "final_adapter")
CANDIDATE_CONTEXT_WINDOW = 80

_CANDIDATE_PATTERN = re.compile(
    r"(?:[A-Z][a-zA-Z']*\s?){1,4}"  # capitalized phrases (names, places)
    r"|[\w.+-]+@[\w.-]+\.\w+"  # emails, in case regex_detector's category config differs
    r"|\b\d[\d\s-]{7,}\b"  # digit-heavy runs (IDs, phone-shaped)
)


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
        out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    try:
        return json.loads(strip_fences(decoded.strip()))
    except json.JSONDecodeError:
        return {"sensitive": False, "category": "other", "confidence": 0.0}


def detect_sensitive_spans(text: str, already_found: List[TextSpan]) -> List[dict]:
    """Same interface/contract as llm_detector.detect_sensitive_spans."""
    already_texts = {s.text.lower() for s in already_found}
    seen = set()
    results = []

    for m in _CANDIDATE_PATTERN.finditer(text):
        candidate = m.group(0).strip()
        if len(candidate) < 3 or candidate.lower() in already_texts or candidate.lower() in seen:
            continue
        seen.add(candidate.lower())

        context = context_window(text, m.start(), m.end(), CANDIDATE_CONTEXT_WINDOW)
        classification = _classify_span(context, candidate)
        if classification.get("sensitive"):
            results.append(
                {
                    "text": candidate,
                    "category": classification.get("category", "other"),
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

        for item in results:
            idx = page_text.find(item["text"])
            if idx == -1:
                continue
            match_start, match_end = idx, idx + len(item["text"])
            covering = [s for (start, end, s) in ranges if start < match_end and end > match_start]
            if not covering:
                continue
            confidence = item["confidence"]
            flagged.append(
                FlaggedSpan(
                    text=item["text"],
                    page_num=page_num,
                    bbox=merge_bbox(covering),
                    span_id=f"local_{page_num}_{idx}",
                    category=item["category"],
                    confidence=confidence,
                    source="local_model",
                    needs_human_review=confidence < CONFIDENCE_THRESHOLD,
                )
            )

    return flagged
