"""True redaction, not an overlay. This is the exact line that differentiates
this project from a fake redactor: `apply_redactions()` strips the underlying
text object, not just draws a black box over it. A `page.draw_rect()` call
looks visually identical but leaves text fully copyable/extractable
underneath - silently producing exactly the bug this project exists to
prevent (and exactly the bug eval's verification-catch-rate metric exists to
prove the verifier catches).

`simulate_failure=True` exists ONLY for eval/run_eval.py's deliberate
leftover-text test cases (Step 3.2 of the guide) - it must never be reachable
from the real pipeline (graph/pipeline_graph.py never passes it).
"""
from typing import List

import pymupdf as fitz

from app.detection.base import FlaggedSpan


def redact_pdf(
    input_path: str,
    output_path: str,
    spans_to_redact: List[FlaggedSpan],
    simulate_failure: bool = False,
    bbox_padding: float = 0.0,
) -> None:
    """bbox_padding grows each redaction rectangle by that many points on
    every side.

    It exists to make the graph's retry edge mean something. Redaction is
    deterministic, so re-running it with identical spans yields an identical
    file and verification fails identically - the retry would just burn a
    cycle. The common real cause of a surviving glyph is a bounding box a
    fraction too tight for the text it covers, so each retry widens the box.
    Over-redacting a neighbouring character is the right trade against
    leaving PII legible.
    """
    doc = fitz.open(input_path)
    for span in spans_to_redact:
        if span.page_num >= len(doc):
            continue
        page = doc[span.page_num]
        x0, y0, x1, y1 = span.bbox
        if bbox_padding:
            x0, y0 = x0 - bbox_padding, y0 - bbox_padding
            x1, y1 = x1 + bbox_padding, y1 + bbox_padding
        page.add_redact_annot((x0, y0, x1, y1), fill=(0, 0, 0))

    if not simulate_failure:
        for page in doc:
            page.apply_redactions()  # THIS actually strips the underlying text
    # else: annotations are added but never applied/removed - PyMuPDF drops
    # unapplied redact annotations silently on save unless we bake in a
    # visible black box ourselves, so simulate the visual-only bug explicitly:
    else:
        for span in spans_to_redact:
            if span.page_num >= len(doc):
                continue
            doc[span.page_num].draw_rect(span.bbox, color=(0, 0, 0), fill=(0, 0, 0))
        for page in doc:
            for annot in list(page.annots() or []):
                if annot.type[0] == 12:  # PDF_ANNOT_REDACT
                    page.delete_annot(annot)

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m app.redaction.redactor <input.pdf> <output.pdf>")
        sys.exit(1)

    from app.extraction.pdf_extractor import extract_pdf
    from app.detection.regex_detector import detect_regex_spans

    doc = extract_pdf(sys.argv[1])
    spans = detect_regex_spans(doc)
    print(f"Redacting {len(spans)} regex-detected spans")
    redact_pdf(sys.argv[1], sys.argv[2], spans)
    print(f"Wrote {sys.argv[2]}")
