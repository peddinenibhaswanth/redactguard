import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api import _validate_upload, app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_rejects_disallowed_extension():
    with pytest.raises(HTTPException) as exc:
        _validate_upload("malware.exe", b"whatever")
    assert exc.value.status_code == 400
    assert "extension" in exc.value.detail


def test_rejects_empty_file():
    with pytest.raises(HTTPException) as exc:
        _validate_upload("doc.pdf", b"")
    assert exc.value.status_code == 400


def test_rejects_content_that_does_not_match_claimed_extension():
    """A file renamed to .pdf that is actually something else (e.g. a plain
    text file, or an executable) must be rejected by magic-byte sniffing,
    not just trusted because of its extension."""
    with pytest.raises(HTTPException) as exc:
        _validate_upload("fake.pdf", b"this is just plain text, not a real PDF")
    assert exc.value.status_code == 400
    assert "doesn't match" in exc.value.detail


def test_accepts_a_real_pdf(sample_pdf):
    with open(sample_pdf, "rb") as f:
        content = f.read()
    ext = _validate_upload("real.pdf", content)
    assert ext == ".pdf"
