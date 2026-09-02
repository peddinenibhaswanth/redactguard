"""Produces a machine-readable JSON report plus a short Markdown summary:
spans redacted (count + categories), spans flagged for human review,
verification result, and how many redact->verify retries were needed.
"""
import json
import os
from typing import List

from app.detection.base import FlaggedSpan


def _spans_by_category(spans: List[FlaggedSpan]) -> dict:
    counts: dict = {}
    for s in spans:
        counts[s.category] = counts.get(s.category, 0) + 1
    return counts


def generate_report(state: dict) -> dict:
    flagged_spans: List[FlaggedSpan] = state.get("flagged_spans", [])
    needs_review = [s for s in flagged_spans if s.needs_human_review]
    verification = state.get("verification_result", {})

    report = {
        "input_path": state.get("input_path"),
        "output_path": state.get("output_path"),
        "file_type": state.get("file_type"),
        "total_spans_redacted": len(flagged_spans),
        "spans_by_category": _spans_by_category(flagged_spans),
        "spans_by_source": {
            src: sum(1 for s in flagged_spans if s.source == src) for src in {s.source for s in flagged_spans}
        },
        "spans_needing_human_review": [
            {"text": s.text, "category": s.category, "confidence": s.confidence, "page": s.page_num}
            for s in needs_review
        ],
        "human_review_needed": len(needs_review) > 0,
        "verification_passed": verification.get("passed"),
        "verification_leftover_spans": verification.get("leftover_spans", []),
        # retry_count counts total redact attempts (starts at 0, incremented
        # inside redact_node on every pass, including the first) - subtract
        # 1 so a clean first-pass document correctly reports 0 retries
        # rather than implying it failed once.
        "redact_attempts": state.get("retry_count", 0),
        "redact_verify_retries": max(0, state.get("retry_count", 0) - 1),
    }
    report["markdown_summary"] = _to_markdown(report)
    return report


def _to_markdown(report: dict) -> str:
    lines = [
        "# Redaction Report",
        "",
        f"- **Input:** {report['input_path']}",
        f"- **Output:** {report['output_path']}",
        f"- **File type:** {report['file_type']}",
        f"- **Spans redacted:** {report['total_spans_redacted']}",
        f"- **By category:** {report['spans_by_category']}",
        f"- **By source:** {report['spans_by_source']}",
        f"- **Verification passed:** {report['verification_passed']}",
        f"- **Redact/verify retries needed:** {report['redact_verify_retries']}",
        f"- **Spans needing human review:** {len(report['spans_needing_human_review'])}",
    ]
    if report["spans_needing_human_review"]:
        lines.append("")
        lines.append("## Needs human review")
        for s in report["spans_needing_human_review"]:
            lines.append(f"- p{s['page']} [{s['category']}] \"{s['text']}\" (confidence={s['confidence']:.2f})")
    return "\n".join(lines)


def save_report(report: dict, out_dir: str, base_name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{base_name}_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    with open(os.path.join(out_dir, f"{base_name}_report.md"), "w", encoding="utf-8") as f:
        f.write(report["markdown_summary"])
