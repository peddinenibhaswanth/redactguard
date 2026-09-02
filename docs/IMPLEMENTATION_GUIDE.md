# RedactGuard — Full Implementation Guide

**Read this whole document before writing any code.** This is written so it can be
pasted into a Copilot/Claude coding session and followed top to bottom without the
assistant needing to guess architecture decisions. Every step says *what* to build,
*why* (so the assistant doesn't "simplify" it into something else), and *what a
correct result looks like*, so mistakes are catchable early instead of three files
downstream.

Build order is **non-negotiable**: Phase 1 fully working → Phase 2 fully working →
only then Phase 3. Do not let the coding assistant start on LangGraph orchestration
or fine-tuning before extraction/redaction/verification work as standalone, testable
functions. Each phase should be runnable and demoable on its own before the next one
starts.

---

## 0. What this project actually is

A pipeline that (1) finds PII in a document more intelligently than plain regex, by
using an LLM for context-dependent spans, and (2) **verifies the redaction actually
removed the text from the underlying file**, instead of trusting the redaction step
blindly. Point (2) is the actual novelty — most toy redaction projects skip it.

Three architectural decisions and why each one is load-bearing, not decorative:

- **LangGraph, not a linear script.** The verification step needs to be able to send
  the document *back* to the redaction step if leftover text is found. A plain
  function pipeline can do this with a `while` loop too — LangGraph is chosen because
  it makes the retry/loop-back explicit as a graph edge with a max-retry guard, which
  is easier to reason about, log, and explain in an interview than nested loop logic.
- **MCP-style tool servers for extraction.** PDF, DOCX, and OCR extraction are
  genuinely different code paths. Wrapping each as its own tool with a uniform
  interface (`extract(file_path) -> ExtractedDocument`) means the graph node calling
  extraction doesn't need to know or care which file type it got. This can be a real
  MCP server, or — if MCP tooling adds friction — a plain Python interface that
  mimics the same contract. **Functionally equivalent; MCP is the "correctly
  packaged" version.** Build the plain interface first, wrap as MCP only once it
  works (see Step 3.2 note).
- **Phased fine-tuning (SFT then DPO), not from the start.** Phase 1 uses an API LLM
  (Gemini/Groq) for the "understands context" detection step. Phase 3 replaces that
  one component with a small locally fine-tuned model. Everything else in the
  pipeline stays identical — this is the point: Phase 3 is a **drop-in swap** of one
  node, not a rewrite.

---

## 1. Repository structure

```
redactguard/
├── app/
│   ├── extraction/
│   │   ├── base.py              # shared ExtractedDocument / TextSpan dataclasses
│   │   ├── pdf_extractor.py     # PyMuPDF-based
│   │   ├── docx_extractor.py    # python-docx-based
│   │   └── ocr_extractor.py     # pytesseract-based (scanned/image PDFs)
│   ├── detection/
│   │   ├── regex_detector.py
│   │   ├── llm_detector.py      # Phase 1: Gemini/Groq call
│   │   └── local_model_detector.py  # Phase 3: fine-tuned model (same interface)
│   ├── redaction/
│   │   └── redactor.py          # PyMuPDF true redaction (not overlay)
│   ├── verification/
│   │   └── verifier.py          # re-extract output, diff against redacted spans
│   ├── graph/
│   │   └── pipeline_graph.py    # LangGraph wiring
│   ├── report/
│   │   └── report_generator.py
│   └── main.py                  # CLI / FastAPI entrypoint
├── eval/
│   ├── generate_synthetic_data.py
│   ├── run_eval.py
│   └── metrics.py
├── phase3_finetune/
│   ├── prepare_sft_dataset.py
│   ├── prepare_dpo_dataset.py
│   ├── colab_sft_train.py
│   ├── colab_dpo_train.py
│   └── export_adapter.py
├── data/
│   ├── synthetic_docs/
│   └── labels/
├── requirements.txt
├── requirements-colab.txt
└── README.md
```

Build files top-to-bottom within each phase's section below. Each module should be
independently unit-testable — the coding assistant should write a quick manual test
(`if __name__ == "__main__":` block or a `tests/` file) for each module before moving
to the next, since silent extraction/coordinate bugs are the hardest thing to debug
once buried inside the graph.

---

## 2. Phase 1 — working pipeline with an API model

### Step 2.1 — Environment setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` (Phase 1 + 2, all free/CPU-only):

```
langgraph>=0.2.0
langchain-core>=0.3.0
pymupdf>=1.24.0
python-docx>=1.1.0
pytesseract>=0.3.10
pillow>=10.0.0
google-generativeai>=0.8.0
groq>=0.11.0
fastapi>=0.112.0
uvicorn>=0.30.0
python-multipart>=0.0.9
pydantic>=2.8.0
python-dotenv>=1.0.0
```

Also install the Tesseract OCR *binary* (not just the Python wrapper) — this is the
single most common setup failure. On Ubuntu/WSL: `sudo apt install tesseract-ocr`.
On Mac: `brew install tesseract`. On Windows: install the UB-Mannheim build and add
it to PATH. Verify with `tesseract --version` before writing any OCR code.

Get free API keys before writing detector code:
- Gemini: https://aistudio.google.com/apikey (generous free tier, already used in
  your AI PDF Chat project — reuse that key)
- Groq: https://console.groq.com/keys (free, fast, good fallback)

Store both in a `.env` file, never hardcoded:
```
GEMINI_API_KEY=...
GROQ_API_KEY=...
```

### Step 2.2 — Extraction layer

**Contract every extractor must satisfy** (`app/extraction/base.py`):

```python
from dataclasses import dataclass
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
    spans: List[TextSpan]
    raw_text: str  # full concatenated text, for regex/LLM passes
```

**Why coordinates matter and can't be skipped:** redaction has to remove text at an
exact location, not just find-and-replace on a text string, because the same string
("John") might appear 20 times and only 3 occurrences are the sensitive one. Every
extractor must return per-occurrence bounding boxes, not just matched strings.

`pdf_extractor.py` — use PyMuPDF's `page.get_text("words")` or `get_text("dict")` to
get word-level bounding boxes, not just `get_text("text")` which discards position.

`docx_extractor.py` — python-docx does not give pixel coordinates (DOCX has no fixed
page layout). Two valid approaches: (a) convert DOCX→PDF first (via `docx2pdf` or
LibreOffice headless: `soffice --headless --convert-to pdf`) then use the PDF
extractor, or (b) track paragraph/run indices instead of coordinates and redact by
replacing run text. **Use approach (a)** — it reuses the PDF redaction logic instead
of duplicating redaction for two coordinate systems. Tell the coding assistant this
explicitly or it will likely build (b) and then hit a wall at the redaction step.

`ocr_extractor.py` — for scanned PDFs/images: `pytesseract.image_to_data()` (not
`image_to_string`) returns per-word bounding boxes and confidence scores in one call,
which is exactly the `TextSpan` shape needed. Route a PDF here only after detecting
it has no extractable text layer (PyMuPDF returns near-empty text on a scanned page —
check `len(page.get_text().strip()) < 20` as the trigger).

### Step 2.3 — Regex detector (`detection/regex_detector.py`)

Fast, cheap, first pass. Cover at minimum: emails, phone numbers (Indian + intl
formats), PAN (`[A-Z]{5}[0-9]{4}[A-Z]`), Aadhaar-shaped 12-digit groups, DIN
(director ID, 8 digits), IFSC codes, credit-card-shaped 16-digit groups, dates of
birth. Return the same `TextSpan` list shape so downstream code doesn't care which
detector found what. Mark every regex hit with `confidence=1.0, source="regex"`.

### Step 2.4 — LLM detector (`detection/llm_detector.py`) — Phase 1's core

This is the component Phase 3 will later replace. Design the interface now so the
swap is trivial later:

```python
def detect_sensitive_spans(text: str, already_found: List[TextSpan]) -> List[dict]:
    """Returns list of {text, span_hint, category, confidence} for spans
    the regex pass missed. Must NOT re-flag anything already in already_found."""
```

Prompt design matters here — this is the part most likely to produce inconsistent
output if under-specified. Require **strict JSON output**, give few-shot examples,
and explicitly instruct the bias described below.

```
System: You are a PII detection specialist reviewing a legal/financial document
for redaction. Regex has already caught structured identifiers (emails, ID
numbers). Your job is to find CONTEXTUAL sensitive information regex would
miss: full names, indirect references to people ("the promoter's spouse"),
addresses, and any domain-specific identifiers.

Rule: if you are not confident whether something is sensitive, mark it
sensitive with confidence < 0.7 rather than omitting it. Missing real PII is
worse than over-flagging — a human will review low-confidence flags.

Return ONLY a JSON array, no prose, no markdown fences:
[{"text": "...", "category": "name|address|indirect_reference|other",
  "confidence": 0.0-1.0}]
```

Call Gemini as primary, Groq (Llama 3.3 70B) as fallback on API failure — you already
have this exact pattern in your AI Research Assistant project, reuse it directly.

**Parsing gotcha:** LLMs frequently wrap JSON in ` ```json ` fences despite
instructions. Strip fences before `json.loads()`:
```python
cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_response.strip())
```
Wrap parsing in try/except with a retry (one re-prompt with "return valid JSON only")
before failing the whole document — don't let one malformed LLM response crash the
pipeline.

**Matching returned text back to coordinates:** the LLM returns text strings, not
coordinates. Match each returned string against `ExtractedDocument.spans` by
substring search within the same page, and if a string appears multiple times, flag
all occurrences (safer default — a missed duplicate is worse than an extra redaction).

### Step 2.5 — Ambiguous-case routing

Any span with `confidence < 0.7` (regex spans are always 1.0, so this only affects
LLM/model spans) gets tagged `needs_human_review = True` in addition to being
redacted. It still gets redacted by default (fail-safe direction), but shows up
separately in the final report so a human can double check false positives weren't
over-redacted. This threshold is a config value, not a magic number buried in code —
put it in a `config.py` or `.env` as `CONFIDENCE_THRESHOLD=0.7`.

### Step 2.6 — Redactor (`redaction/redactor.py`) — the part that must NOT be an overlay

This is the step your original doc correctly flagged as the common failure mode.
Use PyMuPDF's actual redaction annotation API, not `draw_rect`:

```python
import fitz  # PyMuPDF

def redact_pdf(input_path: str, output_path: str, spans_to_redact: list) -> None:
    doc = fitz.open(input_path)
    for span in spans_to_redact:
        page = doc[span.page_num]
        page.add_redact_annot(span.bbox, fill=(0, 0, 0))
    for page in doc:
        page.apply_redactions()  # THIS actually strips the underlying text
    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
```

`apply_redactions()` is what removes the underlying text object, not just draws over
it — this is the exact line that differentiates this project from a fake redactor.
Make sure the coding assistant doesn't substitute a simpler
`page.draw_rect()` call, which looks visually identical but leaves text
extractable underneath — silently producing exactly the bug this project exists to
prevent.

### Step 2.7 — Verifier (`verification/verifier.py`) — the safety net

```python
def verify_redaction(output_path: str, originally_redacted_texts: List[str]) -> dict:
    """Re-extracts text from the OUTPUT file and checks none of the
    originally-flagged sensitive strings are still present."""
    re_extracted = extract_pdf(output_path)  # reuse Step 2.2 extractor
    leftover = [t for t in originally_redacted_texts if t in re_extracted.raw_text]
    return {"passed": len(leftover) == 0, "leftover_spans": leftover}
```

If `passed=False`, the graph (Step 2.8) routes back to redaction. Cap retries at 2 —
if it still fails after 2 attempts, stop and flag the document for manual handling
rather than looping forever (a real failure mode to guard, not a hypothetical one).

### Step 2.8 — LangGraph orchestration (`graph/pipeline_graph.py`)

```python
from langgraph.graph import StateGraph, END

class PipelineState(TypedDict):
    file_path: str
    extracted_doc: ExtractedDocument
    flagged_spans: list
    output_path: str
    verification_result: dict
    retry_count: int

def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("extract", extract_node)
    graph.add_node("regex_detect", regex_detect_node)
    graph.add_node("llm_detect", llm_detect_node)
    graph.add_node("redact", redact_node)
    graph.add_node("verify", verify_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "regex_detect")
    graph.add_edge("regex_detect", "llm_detect")
    graph.add_edge("llm_detect", "redact")
    graph.add_edge("redact", "verify")
    graph.add_conditional_edges(
        "verify",
        lambda state: "report" if state["verification_result"]["passed"]
                      or state["retry_count"] >= 2 else "redact",
        {"report": "report", "redact": "redact"},
    )
    graph.add_edge("report", END)
    return graph.compile()
```

Increment `retry_count` inside `redact_node` itself so the conditional edge has
accurate state. This is the actual "printing press with a QC station that sends work
back" mechanism from your original doc — implemented as a real graph, not a metaphor.

### Step 2.9 — Report generation and entrypoint

Report should be a JSON (machine-readable) + short Markdown summary containing: spans
redacted (count + categories), spans flagged for human review, verification result,
number of redact→verify retries needed. Wrap the graph in a FastAPI endpoint
(`POST /redact`, file upload) and a CLI (`python -m app.main --file doc.pdf`) — CLI
first, since it's faster to debug against during Phase 1/2, add the API wrapper once
the pipeline is stable.

**Phase 1 is done when:** you can run the CLI against a real (test) PDF with fake PII
in it, get a redacted output file, open it in a PDF viewer, and confirm — by
selecting text in the redacted areas — that nothing is copyable. That manual check is
your own personal Phase 1 acceptance test before moving on.

---

## 3. Phase 2 — eval harness

You cannot claim "high precision/recall" or "verification catch rate" on a resume
without having actually measured them. This phase produces the numbers.

### Step 3.1 — Synthetic data generation (`eval/generate_synthetic_data.py`)

You can't use real financial/legal documents (privacy problem, plus you likely don't
have access to real ones with known-ground-truth PII). Generate synthetic ones:

1. Prompt Gemini to generate realistic-sounding fake documents (IPO filing excerpt,
   loan agreement, employment contract) — explicitly instruct it to invent
   plausible-but-fake names, PAN/Aadhaar-shaped numbers, addresses, emails.
2. **Critical:** have the generation prompt also return the ground-truth spans it
   inserted, in a separate structured field, at generation time — don't try to
   re-detect them afterward (that's circular, you'd just be testing your detector
   against itself).

```
Generate a realistic 300-word excerpt from an Indian IPO prospectus. Invent
fake but plausible: 2 director names, 1 PAN number, 1 email, 1 indirect
reference to a person (e.g. "the promoter's spouse"), 1 address.
Return JSON: {"document_text": "...", "ground_truth_spans": [
  {"text": "...", "category": "...", "start_char": N}]}
```

Generate 40-60 documents this way for a small but real eval set. Store as
`data/synthetic_docs/doc_XXX.txt` + `data/labels/doc_XXX.json`.

### Step 3.2 — Deliberate "leftover text" cases for the verifier metric

To actually test the verification step (not just detection), you need documents
where redaction *fails* on purpose, so you can check the verifier catches it. Build a
test mode in the redactor that skips `apply_redactions()` on ~15% of eval runs
(overlay-only, the exact bug this project exists to catch) and confirm the verifier
flags every one of those as `passed=False`. This produces your **verification catch
rate** metric — the number your original doc correctly identified as the
differentiating metric for this project.

### Step 3.3 — Metrics (`eval/metrics.py`, `eval/run_eval.py`)

Standard span-level precision/recall/F1 (a predicted span counts as a match if it
overlaps a ground-truth span by >50% character overlap — exact-match is too strict
for LLM-returned text). Plus:
- **Verification catch rate** = (leftover-text cases correctly flagged) / (total
  deliberately-broken cases)
- **Human-review rate** = fraction of documents that had ≥1 low-confidence flag

Run `run_eval.py` over the full synthetic set, print a results table, save as
`eval/results_phase1.json`. **This file is your resume evidence** — an interviewer
asking "what was your precision/recall" gets a real number, not a guess.

**Phase 2 is done when:** you have `results_phase1.json` with real numbers you can
quote, and you've manually spot-checked 5-10 documents to sanity-check the metric
isn't measuring something wrong (e.g., confirm the overlap-matching logic isn't
trivially inflating recall).

---

## 4. Phase 3 — SFT + DPO fine-tuning (Google Colab)

**This phase does not run on your laptop.** Fine-tuning, even with LoRA, needs a real
GPU. Google Colab's free tier (T4, ~15GB VRAM) is the realistic path. Everything in
this section is written to run in a Colab notebook, with the trained adapter
downloaded afterward and plugged back into your laptop pipeline for CPU inference
(inference of a small LoRA-adapted model on CPU is fine — only training needs GPU).

### Step 4.1 — Why this model, why LoRA/QLoRA

Model: **Qwen2.5-0.5B-Instruct** (or `SmolLM2-1.7B-Instruct` if you want a slightly
larger option and still fit T4 comfortably). Reasoning: small enough that a full
forward+backward pass fits in T4's 15GB with room for DPO's two-model requirement,
already instruction-tuned so SFT needs fewer examples to adapt behavior rather than
teach it from scratch, and small enough for **CPU inference afterward** on your
laptop — this last point matters because it's the actual concrete advantage you're
claiming ("sensitive documents never leave your machine").

LoRA/QLoRA, not full fine-tuning: full fine-tuning of even a 0.5B model in fp32/fp16
plus optimizer states will not reliably fit T4's free-tier VRAM alongside DPO's
reference-model copy. QLoRA (4-bit base model + LoRA adapters) keeps memory low
enough to have real headroom.

### Step 4.2 — Data prep for SFT (`phase3_finetune/prepare_sft_dataset.py`)

Reuse Phase 2's synthetic labeled documents. Convert each ground-truth span (plus
sampled non-sensitive spans as negatives) into an instruction-following format:

```python
SFT_TEMPLATE = """<|im_start|>system
You are a PII detection specialist. Given a text span from a document,
classify whether it contains sensitive information.<|im_end|>
<|im_start|>user
Document context: "{context}"
Span to classify: "{span_text}"<|im_end|>
<|im_start|>assistant
{label_json}<|im_end|>"""
```

Where `label_json` is `{"sensitive": true/false, "category": "...", "confidence": 1.0}`
for clear-cut regex/ground-truth cases (SFT teaches the base pattern), saved as
`data/sft_train.jsonl`. Aim for 300-600 examples minimum — below that, LoRA SFT on a
sub-1B model will not generalize meaningfully; this is a real floor, not a suggestion.
If your 40-60 synthetic documents don't produce enough spans, generate more synthetic
documents in Step 3.1 rather than skimping here — this is the actual bottleneck of
Phase 3, more than compute.

### Step 4.3 — Data prep for DPO (`phase3_finetune/prepare_dpo_dataset.py`)

DPO needs (prompt, chosen, rejected) triples for *ambiguous* cases specifically —
this is where you encode the "prefer flagging when unsure" bias from your original
design:

```python
DPO_EXAMPLE = {
    "prompt": SFT_TEMPLATE_PROMPT_PART.format(context=..., span_text="the promoter's spouse"),
    "chosen": '{"sensitive": true, "category": "indirect_reference", "confidence": 0.6}',
    "rejected": '{"sensitive": false, "category": "none", "confidence": 0.6}',
}
```

Build these from your Phase 2 "indirect reference" and low-confidence ground-truth
categories specifically — DPO is not for the easy/obvious cases (SFT already handles
those), it's for the borderline ones where you want to bias the model's preference.
50-150 pairs is a reasonable target for a first pass. Save as `data/dpo_pairs.jsonl`.

### Step 4.4 — Colab SFT script (`phase3_finetune/colab_sft_train.py`)

Provided as a ready-to-paste-into-Colab-cells script below (Section 5). Uses
`transformers` + `peft` (LoRA) + `trl` (`SFTTrainer`) + `bitsandbytes` (4-bit
quantization). Runtime: Colab menu → Runtime → Change runtime type → **T4 GPU**
before running anything, or every subsequent cell will silently run on CPU and take
hours instead of ~20-40 minutes.

### Step 4.5 — Colab DPO script (`phase3_finetune/colab_dpo_train.py`)

Loads the SFT-adapted checkpoint from Step 4.4, runs `trl`'s `DPOTrainer` on top of
it. DPO must start from the SFT checkpoint, not the base model — this is the correct
order (SFT teaches the task, DPO refines preference on ambiguous cases) and skipping
straight to DPO from the base model will not work well.

### Step 4.6 — Export and integrate

Download the final LoRA adapter (a small folder, tens of MB, not the full model) from
Colab (`files.download()` or save to Google Drive and pull down). Load it locally in
`detection/local_model_detector.py`, which must implement the **exact same interface**
as `llm_detector.py` from Step 2.4 (`detect_sensitive_spans(text, already_found) ->
List[dict]`) — swap it into the LangGraph node in `graph/pipeline_graph.py` in one
line, re-run `eval/run_eval.py` from Phase 2 unchanged, and compare
`results_phase1.json` vs `results_phase3.json`. **This comparison table (API model vs
your fine-tuned model — precision/recall/F1, latency, and "runs entirely offline"
column) is the actual Phase 3 deliverable**, not the training run itself.

**Phase 3 is done when:** `results_phase3.json` exists and you can honestly state
numbers for both models side by side.

---

## 5. Ready-to-use Colab files

Two files are provided alongside this guide:
- `colab_sft_train.py` — paste cell-by-cell into a Colab notebook (split at the `# %%`
  markers) or upload as a `.py` and `!python colab_sft_train.py` it.
- `colab_dpo_train.py` — same, run after SFT completes.

Both assume: Colab T4 runtime, your synthetic datasets already uploaded to
`/content/data/` (drag-and-drop into the Colab file browser, or mount Google Drive).

---

## 6. Common failure points to check for explicitly

Have the coding assistant watch for these — they're the specific ways this project
breaks in practice, not generic advice:

1. **`draw_rect()` instead of `add_redact_annot()` + `apply_redactions()`** in the
   redactor — produces the exact bug this project exists to prevent, and looks
   visually correct so it's easy to miss without the verifier catching it.
2. **DOCX coordinate handling** — do not let the assistant try to invent pixel
   coordinates for DOCX; convert to PDF first (Step 2.2).
3. **LLM JSON parsing** — always strip code fences, always wrap in try/except with
   one retry, never let a malformed response crash the whole document pipeline.
4. **bitsandbytes install on Colab** — occasionally needs `pip install -U
   bitsandbytes` explicitly even after `requirements-colab.txt`, since Colab's
   preinstalled CUDA version can mismatch. If 4-bit loading throws a CUDA error,
   this is the first thing to check.
5. **DPO before SFT** — will produce a poorly-behaved model; enforce the order.
6. **Retry loop with no cap** in the LangGraph conditional edge — always cap at 2
   retries (Step 2.7), or a genuinely unredactable document will loop forever.
7. **Confidence threshold hardcoded in multiple places** — keep it in one config
   value (Step 2.5), since Phase 3's local model will have a different confidence
   calibration than the API model and you'll want to tune this once, not hunt
   through files.
