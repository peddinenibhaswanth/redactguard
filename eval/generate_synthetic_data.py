"""Generates synthetic documents with ground-truth PII spans, so precision/
recall/F1 can be measured against a real answer key instead of re-detecting
and calling that "evaluation" (circular).

Deliberately uses EVAL_GENERATOR_PROVIDER (default: groq) while the detector
uses LLM_DETECTOR_PROVIDER (default: gemini) - generating and detecting with
the SAME model would let the detector's own stylistic patterns leak into the
"ground truth", inflating precision/recall in a way that wouldn't hold up to
an interviewer asking "isn't this eval measuring the model agreeing with
itself?". Different model families keeps the eval honest.

Cannot use real financial/legal documents (privacy problem, and no access to
ones with known ground truth), so this generates plausible fake ones instead
and has the generation prompt return the ground-truth spans it inserted, at
generation time.
"""
import argparse
import glob
import json
import os

from app.config import EVAL_GENERATOR_PROVIDER
from app.llm_client import call_llm, strip_fences

DOC_TYPES = [
    "a 300-word excerpt from an Indian IPO prospectus",
    "a 300-word excerpt from a loan agreement",
    "a 300-word excerpt from an employment contract",
]

SYSTEM_PROMPT = "You generate realistic-but-entirely-fake sample documents for testing a PII redaction tool."

GENERATION_PROMPT_TEMPLATE = """Generate {doc_type}. Invent fake but plausible:
2 director/party names, 1 PAN number (format [A-Z]{{5}}[0-9]{{4}}[A-Z]), 1 email
address, 1 indirect reference to a person (e.g. "the promoter's spouse"), 1
address. Do not use real people, companies, or addresses.

Return ONLY JSON, no prose, no markdown fences, in this exact shape:
{{"document_text": "...", "ground_truth_spans": [
  {{"text": "...", "category": "name|pan|email|indirect_reference|address"}}
]}}
"""


def generate_one(doc_type: str, provider: str = EVAL_GENERATOR_PROVIDER) -> dict:
    prompt = GENERATION_PROMPT_TEMPLATE.format(doc_type=doc_type)
    raw = call_llm(prompt, SYSTEM_PROMPT, provider)
    data = json.loads(strip_fences(raw))

    text = data["document_text"]
    fixed_spans = []
    for span in data.get("ground_truth_spans", []):
        idx = text.find(span["text"])
        if idx == -1:
            continue  # model claimed a span it didn't actually place in the text - drop it
        fixed_spans.append(
            {
                "text": span["text"],
                "category": span.get("category", "other"),
                "start_char": idx,
                "end_char": idx + len(span["text"]),
            }
        )
    return {"document_text": text, "ground_truth_spans": fixed_spans}


def _next_doc_index(out_dir_docs: str) -> int:
    """Highest existing doc_NNN index + 1, so a second run ADDS to the set
    instead of overwriting it. Without this, every run restarts at doc_000
    and silently destroys the documents an existing eval was measured on."""
    existing = glob.glob(os.path.join(out_dir_docs, "doc_*.txt"))
    indices = []
    for path in existing:
        stem = os.path.splitext(os.path.basename(path))[0]
        suffix = stem.removeprefix("doc_")
        if suffix.isdigit():
            indices.append(int(suffix))
    return max(indices) + 1 if indices else 0


def generate_dataset(
    n_docs: int,
    out_dir_docs: str = "data/synthetic_docs",
    out_dir_labels: str = "data/labels",
    provider: str = EVAL_GENERATOR_PROVIDER,
) -> int:
    os.makedirs(out_dir_docs, exist_ok=True)
    os.makedirs(out_dir_labels, exist_ok=True)

    start_index = _next_doc_index(out_dir_docs)
    if start_index:
        print(f"Found {start_index} existing documents - appending, starting at doc_{start_index:03d}")

    generated = 0
    for offset in range(n_docs):
        i = start_index + offset
        doc_type = DOC_TYPES[i % len(DOC_TYPES)]
        try:
            data = generate_one(doc_type, provider)
        except Exception as e:
            print(f"[generate_synthetic_data] skipping doc {i} ({doc_type}): {e}")
            continue

        if not data["ground_truth_spans"]:
            print(f"[generate_synthetic_data] doc {i} had no usable ground-truth spans, skipping")
            continue

        doc_id = f"doc_{i:03d}"
        with open(os.path.join(out_dir_docs, f"{doc_id}.txt"), "w", encoding="utf-8") as f:
            f.write(data["document_text"])
        with open(os.path.join(out_dir_labels, f"{doc_id}.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"doc_id": doc_id, "generator_provider": provider, "ground_truth_spans": data["ground_truth_spans"]},
                f,
                indent=2,
            )
        generated += 1

    print(f"Generated {generated}/{n_docs} synthetic documents using provider={provider}")
    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of synthetic documents to generate.")
    parser.add_argument("--provider", default=EVAL_GENERATOR_PROVIDER, choices=["gemini", "groq"])
    parser.add_argument(
        "--out-docs", default="data/synthetic_docs",
        help="Where to write documents. Use data/heldout_docs to build a test split that "
             "prepare_sft_dataset.py will never train on.",
    )
    parser.add_argument("--out-labels", default="data/labels")
    args = parser.parse_args()
    generate_dataset(args.n, out_dir_docs=args.out_docs, out_dir_labels=args.out_labels, provider=args.provider)
