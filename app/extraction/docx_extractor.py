"""DOCX extraction.

python-docx gives no pixel/point coordinates - DOCX has no fixed page layout,
just flowing paragraphs and runs. Two approaches exist: (a) convert DOCX->PDF
first and reuse the PDF extractor/redactor, or (b) track paragraph/run
indices and redact by replacing run text directly.

This uses (a) deliberately: it reuses the PDF redaction logic (add_redact_annot
+ apply_redactions) instead of building and maintaining a second, different
redaction code path for a second coordinate system. Approach (b) looks
tempting because it skips a conversion step, but it dead-ends at the
redaction step with no coordinate system to redact against.

Conversion tries docx2pdf first (uses MS Word via COM automation on Windows/
Mac - requires Word installed), then falls back to LibreOffice headless
(`soffice --headless --convert-to pdf`) if docx2pdf isn't usable. At least
one of the two must be available on the machine running this.
"""
import os
import shutil
import subprocess
import tempfile

from app.extraction.base import ExtractedDocument
from app.extraction.pdf_extractor import extract_pdf


def _convert_via_docx2pdf(docx_path: str, out_pdf_path: str) -> bool:
    try:
        from docx2pdf import convert
    except ImportError:
        return False
    try:
        convert(docx_path, out_pdf_path)
        return os.path.exists(out_pdf_path)
    except Exception:
        return False


def _convert_via_libreoffice(docx_path: str, out_dir: str) -> str | None:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        return None
    subprocess.run(
        [soffice, "--headless", "--convert-to", "pdf", "--outdir", out_dir, docx_path],
        check=True,
        capture_output=True,
        timeout=120,
    )
    candidate = os.path.join(
        out_dir, os.path.splitext(os.path.basename(docx_path))[0] + ".pdf"
    )
    return candidate if os.path.exists(candidate) else None


def convert_docx_to_pdf(docx_path: str, out_pdf_path: str) -> str:
    if _convert_via_docx2pdf(docx_path, out_pdf_path):
        return out_pdf_path

    libre_result = _convert_via_libreoffice(docx_path, os.path.dirname(out_pdf_path))
    if libre_result:
        if libre_result != out_pdf_path:
            shutil.move(libre_result, out_pdf_path)
        return out_pdf_path

    raise RuntimeError(
        "Could not convert DOCX to PDF: neither docx2pdf (needs MS Word) nor "
        "LibreOffice headless (`soffice`) is available. Install one of them."
    )


def extract_docx(file_path: str) -> ExtractedDocument:
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_pdf = os.path.join(
            tmp_dir, os.path.splitext(os.path.basename(file_path))[0] + ".pdf"
        )
        convert_docx_to_pdf(file_path, out_pdf)
        result = extract_pdf(out_pdf)

    return ExtractedDocument(
        file_path=file_path,
        file_type="docx",
        spans=result.spans,
        raw_text=result.raw_text,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m app.extraction.docx_extractor <path-to-docx>")
        sys.exit(1)

    result = extract_docx(sys.argv[1])
    print(f"file_type={result.file_type}, spans={len(result.spans)}")
    print("Raw text preview:", result.raw_text[:300])
