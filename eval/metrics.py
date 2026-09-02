"""Span-level precision/recall/F1 (overlap-match, not exact-match - LLM-
returned text rarely lines up character-for-character with ground truth, so
a predicted span counts as a true positive if it overlaps a not-yet-matched
ground-truth span by more than 50% of the shorter span's length), plus the
two project-specific metrics: verification catch rate and human-review rate.

Micro-averaged across the whole eval set (sum tp/fp/fn across all docs, then
compute one precision/recall/F1), not averaged per-document - a handful of
docs with zero ground-truth spans would otherwise distort a per-doc average.
"""
from typing import List, TypedDict


class Span(TypedDict):
    text: str
    start_char: int
    end_char: int


def char_overlap_ratio(pred: Span, gt: Span) -> float:
    overlap = max(0, min(pred["end_char"], gt["end_char"]) - max(pred["start_char"], gt["start_char"]))
    shorter = min(pred["end_char"] - pred["start_char"], gt["end_char"] - gt["start_char"])
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def match_spans(predicted: List[Span], ground_truth: List[Span], overlap_threshold: float = 0.5) -> dict:
    matched_gt_idx = set()
    matched_pred_idx = set()

    for pi, p in enumerate(predicted):
        for gi, g in enumerate(ground_truth):
            if gi in matched_gt_idx:
                continue
            if char_overlap_ratio(p, g) > overlap_threshold:
                matched_gt_idx.add(gi)
                matched_pred_idx.add(pi)
                break

    tp = len(matched_pred_idx)
    fp = len(predicted) - tp
    fn = len(ground_truth) - len(matched_gt_idx)
    return {"tp": tp, "fp": fp, "fn": fn}


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def verification_catch_rate(results: List[dict]) -> dict:
    """results: [{"simulated_failure": bool, "verifier_passed": bool}, ...]
    A "catch" is: we deliberately broke redaction AND the verifier correctly
    reported passed=False. A "false alarm" is: redaction was fine but the
    verifier incorrectly reported passed=False."""
    broken_cases = [r for r in results if r["simulated_failure"]]
    clean_cases = [r for r in results if not r["simulated_failure"]]

    caught = sum(1 for r in broken_cases if not r["verifier_passed"])
    false_alarms = sum(1 for r in clean_cases if not r["verifier_passed"])

    return {
        "verification_catch_rate": caught / len(broken_cases) if broken_cases else None,
        "false_alarm_rate": false_alarms / len(clean_cases) if clean_cases else None,
        "broken_cases": len(broken_cases),
        "clean_cases": len(clean_cases),
    }


def human_review_rate(docs_predicted_spans: List[List[dict]], confidence_threshold: float) -> float:
    """Fraction of documents that had >=1 low-confidence flag."""
    if not docs_predicted_spans:
        return 0.0
    flagged_docs = sum(
        1 for spans in docs_predicted_spans if any(s.get("confidence", 1.0) < confidence_threshold for s in spans)
    )
    return flagged_docs / len(docs_predicted_spans)
