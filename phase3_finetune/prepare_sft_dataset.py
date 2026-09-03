"""Converts Phase 2's synthetic labeled documents into SFT instruction
examples: one per ground-truth span (positive, "sensitive": true) plus
sampled non-sensitive spans from the same documents (negative). SFT teaches
the base pattern on clear-cut cases - ambiguous cases are DPO's job
(prepare_dpo_dataset.py), not SFT's.

300-600 examples is a real floor, not a suggestion: below that, LoRA SFT on
a sub-1B model will not generalize meaningfully. If the synthetic doc set
isn't producing enough spans, generate more documents
(eval/generate_synthetic_data.py --n <more>) rather than proceeding with too
few - more synthetic data is the actual bottleneck of Phase 3, more than
compute.
"""
import glob
import json
import os
import random
import re

from phase3_finetune.prompt_template import context_window, format_prompt

MIN_RECOMMENDED_EXAMPLES = 300


def _sample_negative_spans(text: str, ground_truth_spans: list, n: int) -> list:
    claimed = [(s["start_char"], s["end_char"]) for s in ground_truth_spans]
    candidates = []
    for m in re.finditer(r"(?:\S+\s+){0,3}\S+", text):
        start, end = m.start(), m.end()
        if any(start < c_end and end > c_start for c_start, c_end in claimed):
            continue
        candidates.append((start, end, text[start:end]))
    random.shuffle(candidates)
    return candidates[:n]


def build_sft_examples(doc_text: str, label: dict) -> list:
    """Emits {"prompt", "completion"} records rather than one flat "text"
    field. TRL treats a prompt-completion dataset as completion-only loss by
    default, so gradients come only from the short JSON answer.

    With a single "text" field the loss covers the whole sequence, which on
    this task means the model spends nearly all its capacity learning to
    reproduce the long, varied document context and almost none learning the
    answer - training loss plateaus high and generation stays unreliable.
    """
    examples = []
    ground_truth = label["ground_truth_spans"]

    for gt in ground_truth:
        context = context_window(doc_text, gt["start_char"], gt["end_char"])
        label_json = json.dumps({"sensitive": True, "category": gt.get("category", "other"), "confidence": 1.0})
        examples.append({"prompt": format_prompt(context, gt["text"]), "completion": label_json})

    negatives = _sample_negative_spans(doc_text, ground_truth, n=len(ground_truth))
    for start, end, neg_text in negatives:
        context = context_window(doc_text, start, end)
        label_json = json.dumps({"sensitive": False, "category": "none", "confidence": 1.0})
        examples.append({"prompt": format_prompt(context, neg_text), "completion": label_json})

    return examples


def build_dataset(
    docs_dir: str = "data/synthetic_docs",
    labels_dir: str = "data/labels",
    out_path: str = "data/sft_train.jsonl",
) -> int:
    all_examples = []
    for label_path in sorted(glob.glob(os.path.join(labels_dir, "*.json"))):
        doc_id = os.path.splitext(os.path.basename(label_path))[0]
        doc_path = os.path.join(docs_dir, f"{doc_id}.txt")
        if not os.path.exists(doc_path):
            continue
        with open(label_path, encoding="utf-8") as f:
            label = json.load(f)
        with open(doc_path, encoding="utf-8") as f:
            doc_text = f.read()
        all_examples.extend(build_sft_examples(doc_text, label))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(all_examples)} SFT examples to {out_path}")
    if len(all_examples) < MIN_RECOMMENDED_EXAMPLES:
        print(
            f"WARNING: {len(all_examples)} examples is below the ~{MIN_RECOMMENDED_EXAMPLES}-600 floor - "
            f"generate more synthetic documents before training (eval/generate_synthetic_data.py --n <more>)."
        )
    return len(all_examples)


if __name__ == "__main__":
    build_dataset()
