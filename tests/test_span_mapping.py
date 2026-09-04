"""Regression tests for detection -> FlaggedSpan mapping.

The repeated-occurrence case is the important one. local_model_detector used
str.find, which returns only the first match, so a name appearing three times
in a contract produced one FlaggedSpan and the other two occurrences stayed
legible in the "redacted" output - a silent failure of exactly the kind this
project exists to catch.
"""
from app.detection.base import map_results_to_flagged
from app.extraction.base import TextSpan


def _page(words):
    """Builds page_text plus the char-range index the mapper expects."""
    spans, ranges, parts, cursor = [], [], [], 0
    for i, w in enumerate(words):
        s = TextSpan(text=w, page_num=0, bbox=(i * 10.0, 0.0, i * 10.0 + 8.0, 10.0), span_id=f"s{i}")
        spans.append(s)
        ranges.append((cursor, cursor + len(w), s))
        parts.append(w)
        cursor += len(w) + 1
    return " ".join(parts), ranges


def test_every_occurrence_is_flagged_not_just_the_first():
    page_text, ranges = _page(["Paid", "to", "Ravi", "then", "Ravi", "again", "and", "Ravi"])
    results = [{"text": "Ravi", "category": "name", "confidence": 1.0}]

    flagged = map_results_to_flagged(results, page_text, ranges, 0, "llm", 0.7)

    assert len(flagged) == 3, f"expected all 3 occurrences, got {len(flagged)}"
    assert len({f.span_id for f in flagged}) == 3, "span_ids must be unique per occurrence"
    assert {f.bbox for f in flagged}.__len__() == 3, "each occurrence needs its own bbox"


def test_empty_needle_does_not_hang():
    # str.find("", pos) returns pos, so an empty needle would never advance
    # the cursor and the scan loop would spin forever.
    page_text, ranges = _page(["some", "text"])
    flagged = map_results_to_flagged(
        [{"text": "", "category": "other", "confidence": 1.0}], page_text, ranges, 0, "llm", 0.7
    )
    assert flagged == []


def test_low_confidence_is_routed_to_human_review():
    page_text, ranges = _page(["the", "promoter", "spouse"])
    flagged = map_results_to_flagged(
        [{"text": "promoter", "category": "indirect_reference", "confidence": 0.6}],
        page_text, ranges, 0, "llm", 0.7,
    )
    assert len(flagged) == 1
    assert flagged[0].needs_human_review is True


def test_unmatched_text_is_dropped_rather_than_guessed():
    page_text, ranges = _page(["only", "these", "words"])
    flagged = map_results_to_flagged(
        [{"text": "Hallucinated Name", "category": "name", "confidence": 1.0}],
        page_text, ranges, 0, "llm", 0.7,
    )
    assert flagged == [], "text absent from the page must not produce a span"


def test_source_is_recorded_for_attribution():
    page_text, ranges = _page(["Contact", "Ravi", "today"])
    for source in ("llm", "local_model"):
        flagged = map_results_to_flagged(
            [{"text": "Ravi", "category": "name", "confidence": 1.0}],
            page_text, ranges, 0, source, 0.7,
        )
        assert flagged[0].source == source
        assert flagged[0].span_id.startswith(source)
