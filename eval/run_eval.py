"""Runs the full Phase 2 eval: detection precision/recall/F1 against the
synthetic ground truth, human-review rate, and verification catch rate.
Saves eval/results_phase1.json - this file is the resume evidence; an
interviewer asking "what was your precision/recall" gets a real number.

Two different code paths on purpose:
- Detection quality runs regex_detector/llm_detector directly against the
  synthetic .txt documents (fast, no PDF rendering needed - char offsets
  double as the "bbox" x-coordinates so the detectors' existing TextSpan
  contract can be reused unmodified for plain text).
- Verification catch rate runs the REAL redact_pdf()/verify_redaction() on
  REAL rendered PDFs, because that mechanism is what's actually being
  tested (Step 3.2 of the guide) - testing it against fake text spans would
  prove nothing about the actual redaction code path.
"""
import argparse
import glob
import json
import os
import random
import time

import pymupdf as fitz

from app.config import CONFIDENCE_THRESHOLD, LOCAL_ADAPTER_PATH
from app.detection.base import FlaggedSpan, build_page_text_index, merge_bbox, spans_covering_range
from app.detection.llm_detector import detect_llm_spans
from app.detection.regex_detector import detect_regex_spans
from app.extraction.base import ExtractedDocument, TextSpan
from app.extraction.pdf_extractor import extract_pdf
from app.redaction.redactor import redact_pdf
from app.verification.verifier import verify_redaction
from eval.metrics import human_review_rate, match_spans, precision_recall_f1, verification_catch_rate

DATA_DOCS_DIR = "data/synthetic_docs"
DATA_LABELS_DIR = "data/labels"
RESULTS_PATH = "eval/results_phase1.json"
BROKEN_CASE_FRACTION = 0.15


def _text_to_fake_document(text: str, doc_id: str) -> ExtractedDocument:
    """Word-tokenizes on whitespace and uses each word's character offset in
    `text` as its bbox x-range, so regex_detector/llm_detector (which only
    need TextSpan.bbox to merge covering spans, not real geometry) run
    unmodified against plain synthetic text instead of a real PDF."""
    import re

    spans = []
    for i, m in enumerate(re.finditer(r"\S+", text)):
        spans.append(TextSpan(text=m.group(0), page_num=0, bbox=(m.start(), 0, m.end(), 0), span_id=f"{doc_id}_w{i}"))
    return ExtractedDocument(file_path=doc_id, file_type="text", spans=spans, raw_text=text)


def _predicted_spans_with_offsets(text: str, flagged: list) -> list:
    predicted = []
    for span in flagged:
        idx = text.find(span.text)
        if idx == -1:
            continue
        predicted.append(
            {
                "text": span.text,
                "category": span.category,
                "confidence": span.confidence,
                "start_char": idx,
                "end_char": idx + len(span.text),
            }
        )
    return predicted


def _get_detector(name: str):
    """Returns the context-detector to pair with the regex pass. Both satisfy
    the same (doc, already_found) -> List[FlaggedSpan] contract, which is the
    whole point of the Phase 1 / Phase 3 interface being identical."""
    if name == "api":
        return detect_llm_spans
    if name == "local":
        from app.detection.local_model_detector import detect_local_model_spans

        return detect_local_model_spans
    raise ValueError(f"unknown detector {name!r}")


def run_detection_eval(doc_ids: list, detector: str = "api") -> dict:
    detect_context_spans = _get_detector(detector)
    all_tp = all_fp = all_fn = 0
    docs_predicted_spans = []
    per_doc_results = []

    for i, doc_id in enumerate(doc_ids):
        print(f"  [detection {i + 1}/{len(doc_ids)}] {doc_id}", flush=True)
        with open(os.path.join(DATA_DOCS_DIR, f"{doc_id}.txt"), encoding="utf-8") as f:
            text = f.read()
        with open(os.path.join(DATA_LABELS_DIR, f"{doc_id}.json"), encoding="utf-8") as f:
            label = json.load(f)

        fake_doc = _text_to_fake_document(text, doc_id)
        regex_spans = detect_regex_spans(fake_doc)
        context_spans = detect_context_spans(fake_doc, already_found=regex_spans)
        flagged = regex_spans + context_spans

        predicted = _predicted_spans_with_offsets(text, flagged)
        ground_truth = label["ground_truth_spans"]

        match_result = match_spans(predicted, ground_truth)
        all_tp += match_result["tp"]
        all_fp += match_result["fp"]
        all_fn += match_result["fn"]

        docs_predicted_spans.append(predicted)
        per_doc_results.append({"doc_id": doc_id, **match_result})

    overall = precision_recall_f1(all_tp, all_fp, all_fn)
    return {
        "precision": overall["precision"],
        "recall": overall["recall"],
        "f1": overall["f1"],
        "tp": all_tp,
        "fp": all_fp,
        "fn": all_fn,
        "human_review_rate": human_review_rate(docs_predicted_spans, CONFIDENCE_THRESHOLD),
        "per_doc": per_doc_results,
    }


# PyMuPDF's base-14 "helv" font (WinAnsi encoding) does not reliably render
# "smart" punctuation the LLM generator likes to use - when it can't, entire
# characters silently fail to render rather than raising, which can leave a
# page with almost no extractable text. That accidentally routes a plain
# digital-text eval document through the OCR fallback path (pdf_extractor.py
# treats near-empty page text as a scanned page), which then fails if
# Tesseract isn't installed - a confusing failure mode for what should be a
# pure text-rendering step. Normalize to ASCII-safe equivalents instead.
_UNICODE_PUNCT_MAP = str.maketrans(
    {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "--", "…": "..."}
)


def _normalize_punct(text: str) -> str:
    return text.translate(_UNICODE_PUNCT_MAP)


def _render_text_to_pdf(text: str, out_path: str) -> None:
    safe_text = _normalize_punct(text)
    doc = fitz.open()
    page = doc.new_page()
    rect = fitz.Rect(50, 50, page.rect.width - 50, page.rect.height - 50)
    overflow = page.insert_textbox(rect, safe_text, fontsize=10, fontname="helv")
    if overflow < 0:
        print(f"  [warn] text overflowed the page when rendering {out_path} (overflow={overflow:.0f})")
    doc.save(out_path)
    doc.close()


def run_verification_eval(doc_ids: list, out_dir: str = "eval/_tmp_pdfs") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    results = []

    for i, doc_id in enumerate(doc_ids):
        print(f"  [verification {i + 1}/{len(doc_ids)}] {doc_id}", flush=True)
        with open(os.path.join(DATA_DOCS_DIR, f"{doc_id}.txt"), encoding="utf-8") as f:
            text = f.read()
        with open(os.path.join(DATA_LABELS_DIR, f"{doc_id}.json"), encoding="utf-8") as f:
            label = json.load(f)

        pdf_path = os.path.join(out_dir, f"{doc_id}.pdf")
        _render_text_to_pdf(text, pdf_path)

        extracted = extract_pdf(pdf_path)
        page_text, ranges = build_page_text_index(extracted.spans, page_num=0)

        # Redact EVERY occurrence of each ground-truth span's text, not just
        # the first - names and other spans are often mentioned more than
        # once in a document, and a redactor that only catches the first
        # occurrence would leave the rest genuinely (not just simulated-ly)
        # unredacted, which the verifier would then correctly flag as a
        # leftover - that's a real bug in this harness, not a false alarm.
        redact_targets = []
        for gt in label["ground_truth_spans"]:
            needle = _normalize_punct(gt["text"])
            if not needle:
                continue
            search_start = 0
            while True:
                idx = page_text.find(needle, search_start)
                if idx == -1:
                    break
                match_start, match_end = idx, idx + len(needle)
                covering = spans_covering_range(ranges, match_start, match_end)
                if covering:
                    redact_targets.append(
                        FlaggedSpan(
                            text=needle, page_num=0, bbox=merge_bbox(covering), span_id=f"{doc_id}_gt_{idx}",
                            category=gt.get("category", "other"), confidence=1.0, source="ground_truth",
                        )
                    )
                search_start = match_end

        if not redact_targets:
            continue

        simulate_failure = random.random() < BROKEN_CASE_FRACTION
        redacted_path = os.path.join(out_dir, f"{doc_id}_redacted.pdf")
        redact_pdf(pdf_path, redacted_path, redact_targets, simulate_failure=simulate_failure)

        verify_result = verify_redaction(redacted_path, [s.text for s in redact_targets])
        results.append({"doc_id": doc_id, "simulated_failure": simulate_failure, "verifier_passed": verify_result["passed"]})

    return verification_catch_rate(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Cap number of docs (for a quick smoke run).")
    parser.add_argument(
        "--detector", default="api", choices=["api", "local"],
        help="Context detector to pair with regex: 'api' = Gemini/Groq (Phase 1), "
             "'local' = fine-tuned adapter (Phase 3).",
    )
    parser.add_argument(
        "--out", default=None,
        help=f"Where to write results (default {RESULTS_PATH}).",
    )
    parser.add_argument(
        "--skip-verification", action="store_true",
        help="Skip the verification-catch-rate pass. It exercises redact/verify, "
             "which is detector-independent, so it only needs running once.",
    )
    args = parser.parse_args()
    results_path = args.out or RESULTS_PATH

    doc_ids = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(DATA_DOCS_DIR, "*.txt")))
    if args.limit:
        doc_ids = doc_ids[: args.limit]

    if not doc_ids:
        raise SystemExit(
            f"No synthetic docs found in {DATA_DOCS_DIR}. Run "
            f"`python -m eval.generate_synthetic_data --n 50` first."
        )

    print(f"Running eval over {len(doc_ids)} synthetic documents (detector={args.detector})...")

    t0 = time.time()
    detection_results = run_detection_eval(doc_ids, detector=args.detector)
    detection_seconds = time.time() - t0

    verification_results = (
        None if args.skip_verification else run_verification_eval(doc_ids)
    )

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "detector": args.detector,
        "adapter_path": LOCAL_ADAPTER_PATH if args.detector == "local" else None,
        "n_documents": len(doc_ids),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "detection": {k: v for k, v in detection_results.items() if k != "per_doc"},
        "detection_seconds_total": round(detection_seconds, 1),
        "detection_seconds_per_doc": round(detection_seconds / len(doc_ids), 2),
        "verification": verification_results,
        "per_doc_detection": detection_results["per_doc"],
    }

    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nPrecision: {detection_results['precision']:.3f}")
    print(f"Recall:    {detection_results['recall']:.3f}")
    print(f"F1:        {detection_results['f1']:.3f}")
    print(f"Human review rate: {detection_results['human_review_rate']:.3f}")
    print(f"Detection latency: {results['detection_seconds_per_doc']}s/doc")
    if verification_results:
        print(f"Verification catch rate: {verification_results['verification_catch_rate']}")
        print(f"False alarm rate: {verification_results['false_alarm_rate']}")
    print(f"\nSaved {results_path}")


if __name__ == "__main__":
    main()
