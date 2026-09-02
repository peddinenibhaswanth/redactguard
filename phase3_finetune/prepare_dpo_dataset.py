"""Builds (prompt, chosen, rejected) triples for AMBIGUOUS cases specifically
- this is where the "prefer flagging when unsure" bias gets encoded. Built
from Phase 2's indirect-reference ground-truth spans; DPO is not for the
easy/obvious cases (SFT already handles those via prepare_sft_dataset.py).

50-150 pairs is a reasonable target for a first pass. If the synthetic set
doesn't have enough indirect_reference spans, regenerate with more documents
- the generation prompt (eval/generate_synthetic_data.py) always asks for at
least one indirect reference per document.
"""
import glob
import json
import os

from phase3_finetune.prompt_template import context_window, format_prompt

AMBIGUOUS_CATEGORIES = {"indirect_reference"}
MIN_RECOMMENDED_PAIRS = 50


def build_dpo_examples(doc_text: str, label: dict) -> list:
    examples = []
    for gt in label["ground_truth_spans"]:
        if gt.get("category") not in AMBIGUOUS_CATEGORIES:
            continue
        context = context_window(doc_text, gt["start_char"], gt["end_char"])
        prompt = format_prompt(context, gt["text"])
        chosen = json.dumps({"sensitive": True, "category": gt["category"], "confidence": 0.6})
        rejected = json.dumps({"sensitive": False, "category": "none", "confidence": 0.6})
        examples.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return examples


def build_dataset(
    docs_dir: str = "data/synthetic_docs",
    labels_dir: str = "data/labels",
    out_path: str = "data/dpo_pairs.jsonl",
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
        all_examples.extend(build_dpo_examples(doc_text, label))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(all_examples)} DPO pairs to {out_path}")
    if len(all_examples) < MIN_RECOMMENDED_PAIRS:
        print(
            f"WARNING: {len(all_examples)} pairs is below the {MIN_RECOMMENDED_PAIRS}-150 target - "
            f"generate more synthetic documents to raise this before running colab_dpo_train.py."
        )
    return len(all_examples)


if __name__ == "__main__":
    build_dataset()
