"""Real API calls - skipped automatically when no keys are configured (e.g.
in CI, which intentionally does not have GEMINI_API_KEY/GROQ_API_KEY set).
Run locally with a filled-in .env to exercise the actual detection quality.
"""
import os

import pytest

pytestmark = pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY")),
    reason="no LLM API keys configured",
)


def test_llm_detector_finds_name_and_indirect_reference():
    from app.detection.llm_detector import detect_sensitive_spans

    text = (
        "The loan was approved by Rajesh Mehta on behalf of the board. "
        "The property was later transferred to the promoter's spouse."
    )
    results = detect_sensitive_spans(text, already_found=[])

    categories = {r["category"] for r in results}
    assert "name" in categories or "indirect_reference" in categories
    assert len(results) > 0
