from app.detection.regex_detector import detect_regex_spans
from app.extraction.pdf_extractor import extract_pdf


def test_regex_detector_finds_email_phone_and_pan(sample_pdf):
    doc = extract_pdf(sample_pdf)
    flagged = detect_regex_spans(doc)

    categories_found = {s.category for s in flagged}
    assert "email" in categories_found
    assert "phone_indian" in categories_found
    assert "pan" in categories_found

    for span in flagged:
        assert span.source == "regex"
        assert span.confidence == 1.0
        assert span.needs_human_review is False


def test_regex_detector_does_not_double_claim_overlapping_matches(sample_pdf):
    doc = extract_pdf(sample_pdf)
    flagged = detect_regex_spans(doc)

    # PAN "ABCDE1234F" contains a 4-digit run but must be claimed once by
    # the pan pattern, not also re-flagged by a more generic digit pattern.
    pan_hits = [s for s in flagged if s.text == "ABCDE1234F"]
    assert len(pan_hits) == 1
    assert pan_hits[0].category == "pan"
