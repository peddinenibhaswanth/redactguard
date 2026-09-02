from app.extraction.pdf_extractor import extract_pdf


def test_extract_pdf_returns_word_level_spans_with_bbox(sample_pdf):
    doc = extract_pdf(sample_pdf)

    assert doc.file_type == "pdf"
    assert len(doc.spans) > 0
    assert "ramesh.kumar@example.com" in doc.raw_text

    for span in doc.spans:
        assert len(span.bbox) == 4
        x0, y0, x1, y1 = span.bbox
        assert x1 > x0 and y1 >= y0


def test_extract_pdf_finds_the_email_as_a_single_span(sample_pdf):
    doc = extract_pdf(sample_pdf)
    email_spans = [s for s in doc.spans if s.text == "ramesh.kumar@example.com"]
    assert len(email_spans) == 1
