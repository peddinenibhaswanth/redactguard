from app.detection.base import FlaggedSpan
from app.report.report_generator import generate_report


def _make_state(retry_count=1, verification_passed=True):
    flagged_spans = [
        FlaggedSpan("a@b.com", 0, (0, 0, 10, 10), "s1", "email", 1.0, "regex", False),
        FlaggedSpan("indirect ref", 0, (10, 0, 20, 10), "s2", "indirect_reference", 0.6, "llm", True),
    ]
    return {
        "input_path": "in.pdf",
        "output_path": "out.pdf",
        "file_type": "pdf",
        "flagged_spans": flagged_spans,
        "verification_result": {"passed": verification_passed, "leftover_spans": []},
        "retry_count": retry_count,
    }


def test_generate_report_counts_and_categories():
    report = generate_report(_make_state())
    assert report["total_spans_redacted"] == 2
    assert report["spans_by_category"] == {"email": 1, "indirect_reference": 1}
    assert report["spans_by_source"] == {"regex": 1, "llm": 1}
    assert len(report["spans_needing_human_review"]) == 1
    assert report["human_review_needed"] is True


def test_generate_report_first_pass_success_reports_zero_retries():
    """retry_count is incremented on every redact attempt including the
    first, so a clean first-pass document must report 0 retries, not 1."""
    report = generate_report(_make_state(retry_count=1, verification_passed=True))
    assert report["redact_attempts"] == 1
    assert report["redact_verify_retries"] == 0


def test_generate_report_after_one_real_retry():
    report = generate_report(_make_state(retry_count=2, verification_passed=True))
    assert report["redact_attempts"] == 2
    assert report["redact_verify_retries"] == 1


def test_markdown_summary_contains_key_fields():
    report = generate_report(_make_state())
    assert "Redaction Report" in report["markdown_summary"]
    assert "Needs human review" in report["markdown_summary"]
