"""Detection-layer contract layered on top of extraction's TextSpan.

Every detector (regex, LLM, later the fine-tuned local model) returns
FlaggedSpan objects so redaction/report code downstream doesn't need to know
which detector found what - it's the same shape regardless of source.
"""
from dataclasses import dataclass
from typing import List, Tuple

from app.extraction.base import TextSpan


@dataclass
class FlaggedSpan(TextSpan):
    category: str = "other"
    confidence: float = 1.0
    source: str = "regex"  # "regex" | "llm" | "local_model"
    needs_human_review: bool = False


def build_page_text_index(spans: List[TextSpan], page_num: int) -> Tuple[str, List[Tuple[int, int, TextSpan]]]:
    """Joins a page's word-level spans into one string (single space between
    words) and records the character range each span occupies in that
    string, so a regex/substring match on the joined text can be mapped back
    to the one or more word spans (and therefore bboxes) it covers."""
    page_spans = [s for s in spans if s.page_num == page_num]
    parts = []
    ranges = []
    cursor = 0
    for s in page_spans:
        start = cursor
        end = start + len(s.text)
        ranges.append((start, end, s))
        parts.append(s.text)
        cursor = end + 1  # +1 for the joining space
    return " ".join(parts), ranges


def merge_bbox(spans: List[TextSpan]) -> tuple:
    x0 = min(s.bbox[0] for s in spans)
    y0 = min(s.bbox[1] for s in spans)
    x1 = max(s.bbox[2] for s in spans)
    y1 = max(s.bbox[3] for s in spans)
    return (x0, y0, x1, y1)


def spans_covering_range(
    ranges: List[Tuple[int, int, TextSpan]], match_start: int, match_end: int
) -> List[TextSpan]:
    return [s for (start, end, s) in ranges if start < match_end and end > match_start]
