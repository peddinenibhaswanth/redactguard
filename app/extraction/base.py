"""Shared contract every extractor must satisfy. Coordinates are mandatory,
not optional metadata: redaction removes text at an exact location, and the
same string (e.g. "John") can appear many times with only some occurrences
sensitive - so every span needs a per-occurrence bounding box, not just the
matched text."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class TextSpan:
    text: str
    page_num: int
    bbox: tuple  # (x0, y0, x1, y1) in PDF/page coordinate space
    span_id: str  # unique within document


@dataclass
class ExtractedDocument:
    file_path: str
    file_type: str  # "pdf" | "docx" | "scanned_pdf"
    spans: List[TextSpan] = field(default_factory=list)
    raw_text: str = ""  # full concatenated text, for regex/LLM passes
