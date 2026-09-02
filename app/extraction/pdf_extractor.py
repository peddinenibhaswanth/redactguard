"""PyMuPDF-based PDF extraction. Uses get_text("words") for word-level
bounding boxes, never get_text("text") for spans, since plain text discards
position and redaction needs an exact location, not a string match.

Pages with (near) no extractable text layer are routed to OCR automatically -
this is how scanned pages inside an otherwise-digital PDF get handled without
a separate manual step.
"""
import io
import os
from typing import List

import pymupdf as fitz
from PIL import Image

from app.config import OCR_TEXT_LEN_THRESHOLD
from app.extraction.base import ExtractedDocument, TextSpan
from app.extraction.ocr_extractor import extract_words_from_image

OCR_ZOOM = 300 / 72  # render at 300dpi for OCR accuracy vs PDF's native 72dpi


def _extract_page_direct(page, page_num: int, doc_id: str) -> List[TextSpan]:
    words = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, word_no)
    spans = []
    for i, w in enumerate(words):
        x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
        if not text.strip():
            continue
        spans.append(
            TextSpan(
                text=text,
                page_num=page_num,
                bbox=(x0, y0, x1, y1),
                span_id=f"{doc_id}_p{page_num}_w{i}",
            )
        )
    return spans


def _extract_page_ocr(page, page_num: int, doc_id: str) -> List[TextSpan]:
    mat = fitz.Matrix(OCR_ZOOM, OCR_ZOOM)
    pix = page.get_pixmap(matrix=mat)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    return extract_words_from_image(image, page_num, OCR_ZOOM, doc_id)


def extract_pdf(file_path: str) -> ExtractedDocument:
    doc = fitz.open(file_path)
    doc_id = os.path.splitext(os.path.basename(file_path))[0]

    all_spans: List[TextSpan] = []
    raw_text_parts: List[str] = []
    ocr_pages = 0

    for page_num, page in enumerate(doc):
        direct_text = page.get_text().strip()
        if len(direct_text) < OCR_TEXT_LEN_THRESHOLD:
            spans = _extract_page_ocr(page, page_num, doc_id)
            ocr_pages += 1
        else:
            spans = _extract_page_direct(page, page_num, doc_id)

        all_spans.extend(spans)
        raw_text_parts.append(direct_text if direct_text else " ".join(s.text for s in spans))

    file_type = "scanned_pdf" if ocr_pages == len(doc) and len(doc) > 0 else "pdf"
    doc.close()

    return ExtractedDocument(
        file_path=file_path,
        file_type=file_type,
        spans=all_spans,
        raw_text="\n".join(raw_text_parts),
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m app.extraction.pdf_extractor <path-to-pdf>")
        sys.exit(1)

    result = extract_pdf(sys.argv[1])
    print(f"file_type={result.file_type}, spans={len(result.spans)}")
    print("First 5 spans:", result.spans[:5])
    print("Raw text preview:", result.raw_text[:300])
