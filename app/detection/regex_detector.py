"""Fast, cheap, first detection pass. Structured identifiers only - anything
regex can't reliably shape-match (names, indirect references, addresses) is
explicitly left for the LLM pass in llm_detector.py.

Every hit gets confidence=1.0, source="regex" - regex matches are exact by
construction, there's no ambiguity to flag for human review.

Patterns are checked in a fixed priority order and a character range already
claimed by an earlier (more specific) pattern is not re-claimed by a later,
more generic one - e.g. a PAN-shaped match wins over the generic alnum ID
pattern it might otherwise also satisfy.
"""
import re
from typing import Dict, List, Pattern

from app.detection.base import (
    FlaggedSpan,
    build_page_text_index,
    merge_bbox,
    spans_covering_range,
)
from app.extraction.base import ExtractedDocument

# Order matters: most-specific first so generic digit patterns don't steal
# characters a more specific pattern already claimed.
PATTERNS: Dict[str, Pattern] = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "pan": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    "ifsc": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
    "din": re.compile(r"\bDIN[:\s]*\d{8}\b", re.IGNORECASE),
    "credit_card": re.compile(r"\b(?:\d[ -]?){16}\b"),
    "aadhaar": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "phone_indian": re.compile(r"(?<!\d)(?:\+91[-\s]?|0)?[6-9]\d{9}(?!\d)"),
    "phone_intl": re.compile(r"\+\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"),
    "date_of_birth": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
}


def detect_regex_spans(doc: ExtractedDocument) -> List[FlaggedSpan]:
    flagged: List[FlaggedSpan] = []
    page_nums = sorted({s.page_num for s in doc.spans})

    for page_num in page_nums:
        page_text, ranges = build_page_text_index(doc.spans, page_num)
        claimed = [False] * len(page_text)

        for category, pattern in PATTERNS.items():
            for m in pattern.finditer(page_text):
                start, end = m.start(), m.end()
                if any(claimed[start:end]):
                    continue

                covering = spans_covering_range(ranges, start, end)
                if not covering:
                    continue

                for i in range(start, end):
                    claimed[i] = True

                bbox = merge_bbox(covering)
                flagged.append(
                    FlaggedSpan(
                        text=m.group(0),
                        page_num=page_num,
                        bbox=bbox,
                        span_id=f"regex_{page_num}_{start}_{category}",
                        category=category,
                        confidence=1.0,
                        source="regex",
                        needs_human_review=False,
                    )
                )
    return flagged


if __name__ == "__main__":
    from app.extraction.base import ExtractedDocument, TextSpan

    sample = ExtractedDocument(
        file_path="sample.pdf",
        file_type="pdf",
        spans=[
            TextSpan("Contact", 0, (0, 0, 10, 10), "s0"),
            TextSpan("ramesh.kumar@example.com", 0, (10, 0, 30, 10), "s1"),
            TextSpan("or", 0, (30, 0, 35, 10), "s2"),
            TextSpan("9876543210", 0, (35, 0, 50, 10), "s3"),
            TextSpan("PAN:", 0, (50, 0, 55, 10), "s4"),
            TextSpan("ABCDE1234F", 0, (55, 0, 70, 10), "s5"),
        ],
        raw_text="Contact ramesh.kumar@example.com or 9876543210 PAN: ABCDE1234F",
    )
    for f in detect_regex_spans(sample):
        print(f.category, f.text, f.bbox)
