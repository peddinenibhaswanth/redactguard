"""The safety net. Re-extracts text from the OUTPUT file (not the input) and
checks that none of the originally-flagged sensitive strings are still
recoverable. This is the actual novelty of the project: redaction is treated
as "unverified until proven" rather than trusted blindly.

If passed=False, the graph (graph/pipeline_graph.py) routes back to
redaction, capped at MAX_REDACT_RETRIES - a genuinely unredactable document
must stop and get flagged for manual handling, not loop forever.
"""
from typing import List

from app.extraction.pdf_extractor import extract_pdf


def verify_redaction(output_path: str, originally_redacted_texts: List[str]) -> dict:
    re_extracted = extract_pdf(output_path)
    leftover = [t for t in originally_redacted_texts if t and t in re_extracted.raw_text]
    return {"passed": len(leftover) == 0, "leftover_spans": leftover}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m app.verification.verifier <redacted.pdf> <text1> [text2 ...]")
        sys.exit(1)

    result = verify_redaction(sys.argv[1], sys.argv[2:])
    print(result)
