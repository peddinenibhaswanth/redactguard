"""The safety net. Re-extracts text from the OUTPUT file (not the input) and
checks that none of the originally-flagged sensitive strings are still
recoverable. This is the actual novelty of the project: redaction is treated
as "unverified until proven" rather than trusted blindly.

If passed=False, the graph (graph/pipeline_graph.py) routes back to
redaction, capped at MAX_REDACT_RETRIES - a genuinely unredactable document
must stop and get flagged for manual handling, not loop forever.
"""
import re
from typing import List

from app.extraction.pdf_extractor import extract_pdf


def _normalize(text: str) -> str:
    """Collapses runs of whitespace so re-extraction artefacts cannot hide a
    leftover.

    An exact substring test compares the flagged string against text
    re-extracted from a different file, and PDF extraction does not guarantee
    identical spacing - a name that comes back as "Rajesh  Mehta" or split
    across a line break would not match "Rajesh Mehta", and the verifier would
    report PASSED while the PII sat there readable. Normalising errs toward
    detecting leftovers, which is the correct direction for a safety net: a
    false alarm costs one capped retry, a miss ships unredacted PII.
    """
    return re.sub(r"\s+", " ", text).strip()


def verify_redaction(output_path: str, originally_redacted_texts: List[str]) -> dict:
    """Note the limitation: this checks whole flagged strings. If redaction
    removed "Mehta" but left "Rajesh" behind, the full string no longer
    matches and this reports passed. Catching partial leftovers would need
    token-level checking - worth doing, not done here.
    """
    re_extracted = extract_pdf(output_path)
    haystack = _normalize(re_extracted.raw_text)
    leftover = [t for t in originally_redacted_texts if t and _normalize(t) in haystack]
    return {"passed": len(leftover) == 0, "leftover_spans": leftover}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m app.verification.verifier <redacted.pdf> <text1> [text2 ...]")
        sys.exit(1)

    result = verify_redaction(sys.argv[1], sys.argv[2:])
    print(result)
