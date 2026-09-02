from app.detection.regex_detector import detect_regex_spans
from app.extraction.pdf_extractor import extract_pdf
from app.redaction.redactor import redact_pdf
from app.verification.verifier import verify_redaction


def _flagged_spans(sample_pdf):
    doc = extract_pdf(sample_pdf)
    return detect_regex_spans(doc)


def test_true_redaction_removes_text_and_verifier_confirms_it(sample_pdf, tmp_path):
    flagged = _flagged_spans(sample_pdf)
    out_path = str(tmp_path / "redacted.pdf")

    redact_pdf(sample_pdf, out_path, flagged)
    result = verify_redaction(out_path, [s.text for s in flagged])

    assert result["passed"] is True
    assert result["leftover_spans"] == []


def test_verifier_catches_overlay_only_fake_redaction(sample_pdf, tmp_path):
    """This is the project's core claim: draw_rect()-style overlay redaction
    LOOKS correct but leaves text extractable - the verifier must catch it."""
    flagged = _flagged_spans(sample_pdf)
    out_path = str(tmp_path / "broken.pdf")

    redact_pdf(sample_pdf, out_path, flagged, simulate_failure=True)
    result = verify_redaction(out_path, [s.text for s in flagged])

    assert result["passed"] is False
    assert set(result["leftover_spans"]) == {s.text for s in flagged}
