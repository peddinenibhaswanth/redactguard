"""OCR extraction for scanned/image PDF pages, via pytesseract.

Uses image_to_data() (not image_to_string()) because it returns per-word
bounding boxes and confidence scores in one call - exactly the TextSpan shape
needed. Requires the Tesseract OCR *binary* to be installed and on PATH, not
just the pytesseract wrapper - verify with `tesseract --version` before
debugging "why is OCR returning nothing".
"""
from typing import List

import pytesseract
from PIL import Image

from app.extraction.base import TextSpan


def extract_words_from_image(
    image: Image.Image,
    page_num: int,
    zoom: float,
    span_id_prefix: str,
    min_confidence: int = 30,
) -> List[TextSpan]:
    """image was rendered at `zoom`x the PDF's native 72dpi space, so pixel
    coordinates from pytesseract must be divided by `zoom` to land back in
    PDF point space, where redaction bboxes are expected."""
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    spans: List[TextSpan] = []
    n = len(data["text"])
    for i in range(n):
        word = data["text"][i].strip()
        if not word:
            continue
        try:
            conf = int(float(data["conf"][i]))
        except (ValueError, TypeError):
            conf = -1
        if conf < min_confidence:
            continue

        left, top, width, height = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )
        bbox = (left / zoom, top / zoom, (left + width) / zoom, (top + height) / zoom)
        spans.append(
            TextSpan(
                text=word,
                page_num=page_num,
                bbox=bbox,
                span_id=f"{span_id_prefix}_ocr_p{page_num}_w{i}",
            )
        )
    return spans
