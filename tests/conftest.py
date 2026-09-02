import pymupdf
import pytest


@pytest.fixture
def sample_pdf(tmp_path):
    """A small PDF with regex-catchable PII, for extraction/redaction/
    verification tests that don't need real geometry beyond 'is on page 0'."""
    path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    text = "Contact Ramesh Kumar at ramesh.kumar@example.com or 9876543210. PAN: ABCDE1234F."
    page.insert_text((50, 72), text, fontsize=12)
    doc.save(str(path))
    doc.close()
    return str(path)
