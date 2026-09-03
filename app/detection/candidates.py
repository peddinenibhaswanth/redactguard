"""Candidate span generation for the fine-tuned local detector.

Single source of truth on purpose. The fine-tuned model is a *classifier* over
(context, span) pairs - it cannot propose spans itself - so something has to
hand it candidates at inference time. If the negatives it trains on are drawn
from a different distribution than the candidates it is later asked about, it
never learns to reject the things it will actually see.

That was a real, measured failure: SFT negatives were random sentence
fragments ("business focuses on developing") while inference candidates are
capitalised entity phrases ("Suryam Tech Solutions Ltd", "Companies Act").
The model had no signal distinguishing a company name from a person's name and
flagged nearly everything - 279 false positives across 12 documents, precision
0.182.

Both phase3_finetune/prepare_sft_dataset.py and
app/detection/local_model_detector.py import from here so training and
inference cannot drift apart again.
"""
import re
from typing import Iterator, Tuple

CANDIDATE_PATTERN = re.compile(
    r"(?:[A-Z][a-zA-Z']*\s?){1,4}"  # capitalised phrases (names, orgs, places)
    r"|[\w.+-]+@[\w.-]+\.\w+"  # emails
    r"|\b\d[\d\s-]{7,}\b"  # digit-heavy runs (IDs, phone-shaped)
)

MIN_CANDIDATE_LEN = 3


def iter_candidates(text: str) -> Iterator[Tuple[int, int, str]]:
    """Yields (start, end, span_text) for each deduplicated candidate."""
    seen = set()
    for m in CANDIDATE_PATTERN.finditer(text):
        span = m.group(0).strip()
        key = span.lower()
        if len(span) < MIN_CANDIDATE_LEN or key in seen:
            continue
        seen.add(key)
        yield m.start(), m.start() + len(span), span
