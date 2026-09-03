"""Phase 1's core detector - the component Phase 3 replaces with a fine-tuned
local model. `detect_sensitive_spans()` is the exact interface Phase 3's
`local_model_detector.py` must also implement, so the swap in the LangGraph
node is a one-line change, not a rewrite.

Calls Gemini as primary, Groq (Llama 3.3 70B) as fallback on API failure.

Long documents are chunked per page into overlapping windows before being
sent to the LLM - a single call over a 50-page filing would blow past
context limits and rate limits, so pages beyond a size threshold are split
rather than sent whole.
"""
import json
from typing import List

from app.config import CONFIDENCE_THRESHOLD, LLM_DETECTOR_PROVIDER
from app.detection.base import FlaggedSpan, build_page_text_index, merge_bbox
from app.extraction.base import ExtractedDocument, TextSpan
from app.llm_client import call_llm, strip_fences

SYSTEM_PROMPT = """You are a PII detection specialist reviewing a legal/financial document
for redaction. Regex has already caught structured identifiers (emails, ID
numbers). Your job is to find CONTEXTUAL sensitive information regex would
miss: full names, indirect references to people ("the promoter's spouse"),
addresses, and any domain-specific identifiers.

Rule: if you are not confident whether something is sensitive, mark it
sensitive with confidence < 0.7 rather than omitting it. Missing real PII is
worse than over-flagging - a human will review low-confidence flags.

Examples:
Text: "The loan was approved by Rajesh Mehta on behalf of the board."
Output: [{"text": "Rajesh Mehta", "category": "name", "confidence": 0.95}]

Text: "The property was transferred to the promoter's spouse last year."
Output: [{"text": "the promoter's spouse", "category": "indirect_reference", "confidence": 0.6}]

Text: "Payment is due within 30 days of invoice."
Output: []

Return ONLY a JSON array, no prose, no markdown fences:
[{"text": "...", "category": "name|address|indirect_reference|other", "confidence": 0.0-1.0}]
"""

MAX_CHUNK_CHARS = 3000
CHUNK_OVERLAP = 200


def _chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _parse_json_with_retry(raw: str, prompt: str, provider: str) -> list:
    cleaned = strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    retry_prompt = prompt + "\n\nYour previous response was not valid JSON. Return valid JSON only."
    try:
        retry_raw = call_llm(retry_prompt, SYSTEM_PROMPT, provider)
        return json.loads(strip_fences(retry_raw))
    except json.JSONDecodeError:
        # Log it. Returning [] silently makes a chunk with unparseable output
        # indistinguishable from a chunk that genuinely contained no PII - a
        # 67-document eval scored recall 0.43 on one batch before this was
        # visible, and the cause looked like bad data rather than a failed
        # API call.
        print(f"[llm_detector] unparseable JSON after retry, treating chunk as empty: {retry_raw[:120]!r}")
        return []
    except Exception as e:
        print(f"[llm_detector] retry call failed, treating chunk as empty: {e}")
        return []


def detect_sensitive_spans(
    text: str, already_found: List[TextSpan], provider: str = LLM_DETECTOR_PROVIDER
) -> List[dict]:
    """Returns list of {text, category, confidence} for spans the regex pass
    missed. Must NOT re-flag anything already in already_found."""
    if not text.strip():
        return []

    already_texts = {s.text.lower() for s in already_found}
    results = []

    for chunk in _chunk_text(text):
        prompt = f'Document excerpt:\n"""\n{chunk}\n"""'
        try:
            raw = call_llm(prompt, SYSTEM_PROMPT, provider)
        except Exception as e:
            print(f"[llm_detector] both providers failed on a chunk: {e}")
            continue

        parsed = _parse_json_with_retry(raw, prompt, provider)
        if not isinstance(parsed, list):
            continue

        for item in parsed:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
                continue
            if item["text"].strip().lower() in already_texts:
                continue
            results.append(
                {
                    "text": item["text"],
                    "category": item.get("category", "other"),
                    "confidence": float(item.get("confidence", 0.5)),
                }
            )

    return results


def detect_llm_spans(
    doc: ExtractedDocument, already_found: List[FlaggedSpan], provider: str = LLM_DETECTOR_PROVIDER
) -> List[FlaggedSpan]:
    """Runs detect_sensitive_spans() per page and matches the returned text
    strings back to coordinates by substring search within that page. If a
    string appears multiple times on a page, all occurrences are flagged -
    a missed duplicate is worse than an extra redaction."""
    flagged: List[FlaggedSpan] = []
    page_nums = sorted({s.page_num for s in doc.spans})

    for page_num in page_nums:
        page_text, ranges = build_page_text_index(doc.spans, page_num)
        page_already_found = [s for s in already_found if s.page_num == page_num]

        llm_results = detect_sensitive_spans(page_text, page_already_found, provider)

        for item in llm_results:
            needle = item["text"]
            if not needle:
                continue  # str.find("", pos) always returns pos - would spin forever below
            search_start = 0
            occurrence = 0
            while True:
                idx = page_text.find(needle, search_start)
                if idx == -1:
                    break
                match_start, match_end = idx, idx + len(needle)
                covering = [s for (start, end, s) in ranges if start < match_end and end > match_start]
                if covering:
                    bbox = merge_bbox(covering)
                    confidence = item["confidence"]
                    flagged.append(
                        FlaggedSpan(
                            text=needle,
                            page_num=page_num,
                            bbox=bbox,
                            span_id=f"llm_{page_num}_{idx}_{occurrence}",
                            category=item["category"],
                            confidence=confidence,
                            source="llm",
                            needs_human_review=confidence < CONFIDENCE_THRESHOLD,
                        )
                    )
                search_start = match_end
                occurrence += 1

    return flagged


if __name__ == "__main__":
    import sys

    sample_text = sys.argv[1] if len(sys.argv) > 1 else (
        "The loan was approved by Rajesh Mehta on behalf of the board. "
        "The property was later transferred to the promoter's spouse."
    )
    print(detect_sensitive_spans(sample_text, already_found=[]))
