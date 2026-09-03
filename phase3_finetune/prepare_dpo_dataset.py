"""Builds (prompt, chosen, rejected) triples for AMBIGUOUS cases - the place
where the "prefer flagging when unsure" bias gets encoded. DPO is not for the
easy/obvious cases; SFT already handles those (prepare_sft_dataset.py).

Pairs run in BOTH directions, deliberately:

  positive direction - a real indirect reference ("the promoter's spouse")
      chosen   = sensitive true
      rejected = sensitive false

  negative direction - a reference-shaped span that is NOT a person
      ("the Company", "the Board of Directors", "the Lender")
      chosen   = sensitive false
      rejected = sensitive true

The second group is what makes this a real preference task. With only the
positive direction every pair shares one identical chosen string and one
identical rejected string, so the optimal policy is "always answer true"
without ever reading the prompt - DPO then converges in a handful of steps to
accuracies=1.0 while learning nothing transferable, and precision collapses at
eval time. Mixing directions forces the decision to depend on the input.

In practice this lands near a 1:1 split: most documents contain exactly one
indirect reference, and the per-document floor of one contrast span means
each positive is matched by roughly one negative. The fail-safe lean toward
flagging is carried by the SFT stage's system prompt rather than by skewing
these counts.
"""
import glob
import json
import os
import random
import re

from phase3_finetune.prompt_template import context_window, format_prompt

AMBIGUOUS_CATEGORIES = {"indirect_reference"}
MIN_RECOMMENDED_PAIRS = 50
NEGATIVE_RATIO = 0.5  # negative-direction pairs per positive-direction pair
RANDOM_SEED = 13

# Reference-shaped spans: "the <Capitalised Entity>" - "the Company",
# "the Employee", "the Board of Directors", "the Companies Act". These are
# genuinely confusable with a personal indirect reference, which is the point;
# an easy negative teaches nothing.
#
# Capitalisation is required deliberately. A lowercase alternative here also
# matched verb phrases ("the parties have executed", "the entire understanding
# between") - sentence fragments no detector would ever consider PII, so they
# make the contrast trivially easy instead of instructive.
_REFERENCE_SHAPED = re.compile(
    r"\bthe\s+[A-Z][a-zA-Z]+(?:\s+(?:of\s+)?[A-Z][a-zA-Z]+){0,3}"
)


def _overlaps_ground_truth(start: int, end: int, ground_truth: list) -> bool:
    return any(start < gt["end_char"] and end > gt["start_char"] for gt in ground_truth)


def build_dpo_examples(doc_text: str, label: dict, rng: random.Random) -> list:
    ground_truth = label["ground_truth_spans"]
    examples = []

    positives = 0
    for gt in ground_truth:
        if gt.get("category") not in AMBIGUOUS_CATEGORIES:
            continue
        context = context_window(doc_text, gt["start_char"], gt["end_char"])
        examples.append(
            {
                "prompt": format_prompt(context, gt["text"]),
                "chosen": json.dumps({"sensitive": True, "category": gt["category"], "confidence": 0.6}),
                "rejected": json.dumps({"sensitive": False, "category": "none", "confidence": 0.6}),
            }
        )
        positives += 1

    if not positives:
        return examples

    # Contrast set: reference-shaped spans that are not ground-truth PII.
    candidates = []
    for m in _REFERENCE_SHAPED.finditer(doc_text):
        start, end = m.start(), m.end()
        text = m.group(0).strip()
        if len(text) < 6 or _overlaps_ground_truth(start, end, ground_truth):
            continue
        candidates.append((start, end, text))

    rng.shuffle(candidates)
    for start, end, text in candidates[: max(1, round(positives * NEGATIVE_RATIO))]:
        context = context_window(doc_text, start, end)
        examples.append(
            {
                "prompt": format_prompt(context, text),
                "chosen": json.dumps({"sensitive": False, "category": "none", "confidence": 0.6}),
                "rejected": json.dumps({"sensitive": True, "category": "indirect_reference", "confidence": 0.6}),
            }
        )

    return examples


def build_dataset(
    docs_dir: str = "data/synthetic_docs",
    labels_dir: str = "data/labels",
    out_path: str = "data/dpo_pairs.jsonl",
) -> int:
    rng = random.Random(RANDOM_SEED)
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
        all_examples.extend(build_dpo_examples(doc_text, label, rng))

    rng.shuffle(all_examples)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    n_flag = sum(1 for e in all_examples if json.loads(e["chosen"])["sensitive"])
    print(f"Wrote {len(all_examples)} DPO pairs to {out_path}")
    print(f"  prefer-flag pairs   : {n_flag}")
    print(f"  prefer-ignore pairs : {len(all_examples) - n_flag}")
    print(f"  distinct chosen strings: {len({e['chosen'] for e in all_examples})} "
          f"(1 would mean the task is degenerate)")
    if len(all_examples) < MIN_RECOMMENDED_PAIRS:
        print(
            f"WARNING: {len(all_examples)} pairs is below the {MIN_RECOMMENDED_PAIRS}-150 target - "
            f"generate more synthetic documents before running colab_dpo_train.py."
        )
    return len(all_examples)


if __name__ == "__main__":
    build_dataset()
