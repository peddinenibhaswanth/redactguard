# RedactGuard — Complete Project Report

A full account of the project from first review to final results: what was
built, what broke, what each failure turned out to be, and every number
measured along the way.

**Repository:** https://github.com/peddinenibhaswanth/redactguard
**Live demo:** https://redactguard-66iybemnkbkxq6ngdvxqkx.streamlit.app/
**Built:** 2 – 4 September 2026 · 26 commits · 77 files · ~3,300 lines of Python

Every metric in this document traces to a committed JSON file in `eval/`.
Nothing here is estimated.

---

## 1. What the project is

A pipeline that removes personally identifiable information (PII) from PDF and
DOCX documents — and then **verifies its own work** by re-extracting the
output file and confirming the sensitive text is genuinely gone.

### The problem it targets

A naive PDF redactor draws a black rectangle over sensitive text. The file
*looks* redacted. The text underneath is still there — selectable, copyable,
extractable by anyone who opens it.

In PyMuPDF the difference is one line:

```python
page.draw_rect(bbox, fill=(0,0,0))                  # looks right, text survives
page.add_redact_annot(bbox); page.apply_redactions()  # actually strips the text
```

Both produce files that are visually identical. That is what makes it
dangerous: the failure is invisible.

**The project's core contribution** is treating its own redaction as
unverified until proven. After redacting, it re-opens the output file,
re-extracts the text, and checks that none of the originally-flagged strings
survive. If any do, the document is routed back through redaction
automatically, capped at a retry limit, then flagged for manual handling if it
still fails.

---

## 2. Architecture

```
extract → regex_detect → llm_detect → redact → verify → report
                                        ↑         │
                                        └─────────┘
                                    (verification failed,
                                     retries remaining)
```

| Layer | Module | Responsibility |
|---|---|---|
| Extraction | `app/extraction/` | PyMuPDF for PDFs (word-level bounding boxes), python-docx → PDF for DOCX, pytesseract for scanned pages |
| Detection | `app/detection/` | Regex pass for structured IDs; LLM pass for context-dependent PII |
| Redaction | `app/redaction/` | True PyMuPDF redaction (`apply_redactions`) |
| Verification | `app/verification/` | Re-extracts output, diffs against flagged text |
| Orchestration | `app/graph/` | LangGraph `StateGraph`; verification failure is a real graph edge |
| Reporting | `app/report/` | JSON + Markdown reports |

**Detection is two-tier by design.** Regex handles what has reliable shape —
emails, PAN, Aadhaar-format numbers, IFSC codes, Indian phone numbers. The LLM
handles what has no shape: personal names, street addresses, and *indirect
references* like "the promoter's spouse", which identify a person without
naming them. Regex cannot express that; a language model can.

**Why LangGraph rather than a `while` loop.** The retry is a conditional graph
edge, which makes control flow inspectable — state visible at each node, retry
count loggable, loop explainable without reading nested control flow. It also
made the Phase 3 detector swap a one-import change.

### Tech stack

Python · LangGraph · PyMuPDF · python-docx · pytesseract · Gemini API · Groq
API · FastAPI · Streamlit · Docker · pytest · GitHub Actions · Hugging Face
Transformers · PEFT · TRL · PyTorch

---

## 3. Timeline of what actually happened

### Stage 0 — Plan review

The project began as a written implementation guide. Reviewing it before any
code, I flagged eight gaps:

1. No demo UI — the single biggest gap for a portfolio piece
2. Eval bias risk: generating test data and detecting with the same model family
3. No tests, CI, or Docker
4. Long-document handling unaddressed (eval set is ~300-word excerpts)
5. Package pins ~2 years stale
6. DOCX→PDF conversion needed a Windows-specific decision
7. No input validation on the API endpoint — ironic for a security-adjacent tool
8. "Documents never leave your machine" only true of the Phase 3 configuration

All eight were addressed except (4), which is documented as a limitation.

### Stage 1 — Core pipeline

Built extraction, detection, redaction, verification, reporting, and the
LangGraph orchestration. Added a FastAPI service with upload validation, a
Dockerfile, a pytest suite, and GitHub Actions CI.

### Stage 2 — Evaluation harness

Synthetic documents are generated **with ground truth attached at creation
time**, not re-detected afterwards — re-detecting and calling it evaluation
would just test the detector against itself. Generation uses Groq while
detection uses Gemini, so the eval is not one model agreeing with itself.

The harness measures precision, recall, F1, human-review rate, and the
project's differentiating metric: **verification catch rate**.

**How the verifier is validated:** overlay-only ("fake") redactions are
deliberately injected into ~15% of eval runs via
`redact_pdf(..., simulate_failure=True)`. The metric is how many of those
deliberate failures the verifier catches.

**First result (37 documents):** precision 0.798 · recall 0.946 · **F1 0.865**
· verification catch rate **1.000** (7/7 caught) · false alarm rate **0.000**

### Stage 3 — Deployment (three attempts)

**Attempt 1 — Hugging Face Spaces.** Failed. My deployment instructions told
the user to flatten `demo/app.py` to the repo root, which breaks its import
of the `app` package. Caught before deploying and corrected to preserve the
folder structure.

**Attempt 2 — Hugging Face Spaces, again.** Failed for a different reason:
Hugging Face no longer offers free compute-backed Spaces. Only Static is free;
Gradio and Docker require PRO, and Streamlit is no longer an SDK option at
all. The Space was accepted but sat permanently **Paused**.

**Attempt 3 — Streamlit Community Cloud.** Succeeded. Two changes were needed:

- **Secrets bridge.** Streamlit Cloud supplies secrets via `st.secrets`, but
  `app/config.py` reads environment variables *at import time*. Without
  bridging before that import, keys read as empty and detection silently
  degraded to regex-only.
- **Dependency split.** `torch`/`transformers`/`peft` (~2 GB) moved to
  `requirements-phase3.txt`; they are only needed for optional local
  inference and would have exceeded the host's resource limit.

**A user error worth recording.** The clone command was run without a target
directory, so it cloned into `redactguard/` instead of `space-repo/`.
`Copy-Item` then created `space-repo/` as a plain folder, and `git add` inside
a non-git directory walked *up* and committed junk into the main project repo.
Recovered by rewinding the commit, deleting the stray files, and gitignoring
clone directories so it cannot recur.

**A mistake of mine worth recording.** I probed the deployed URL with bare
`curl`, got a 303 redirect to a login path, and told the user their app was
private. It was not — `curl` without cookie handling was stuck mid-handshake.
Re-testing with a cookie jar returned HTTP 200. The user's incognito test had
been correct and I should have trusted it over my flawed probe.

### Stage 4 — Sample document

Created `samples/generate_sample.py`, producing a synthetic loan-agreement
extract with entirely invented identifiers (`example.com` emails, the standard
dummy PAN pattern, placeholder Aadhaar digits). Shaped so one upload exercises
the whole pipeline.

Verified end-to-end: **9 spans redacted** (5 regex + 4 LLM), all 8 categories
hit, verification passed on the first attempt, and exactly one span — *"the
promoter's spouse"* at confidence 0.60 — routed to human review.

### Stage 5 — Phase 3 fine-tuning

The goal: replace the API detection node with a locally fine-tuned model
behind an **identical interface**, so the swap is one import change. The
deliverable was always the *comparison*, not the training run.

**Model:** Qwen2.5-0.5B-Instruct · **Method:** QLoRA (4-bit NF4 base, LoRA
r=16 on attention projections) · **Hardware:** free-tier Colab T4

This stage took **seven distinct debugging rounds**. Each is worth
understanding.

---

## 4. The seven Phase 3 failures, and what each actually was

### Failure 1 — Stale library API

The provided training scripts pinned mid-2024 versions and used APIs that no
longer exist in TRL 1.x. Three would have thrown `TypeError` on trainer
construction:

| Script used | Current API |
|---|---|
| `tokenizer=` | `processing_class=` |
| `max_seq_length` | `max_length` (SFTConfig) |
| `max_prompt_length` | removed (DPOConfig) |

Found by introspecting the installed library rather than trusting the pins.

### Failure 2 — bf16 on hardware that cannot do bf16

Both scripts hardcoded `bf16=True`. **A T4 is Turing (compute capability 7.5)
and has no bfloat16 support at all** — bf16 requires Ampere or newer. Fixed by
detecting support at runtime via `torch.cuda.is_bf16_supported()` and falling
back to fp16, for both the trainer and the 4-bit compute dtype.

### Failure 3 — Missing QLoRA stabilisation

The scripts never called `prepare_model_for_kbit_training`, which casts
layernorms and upcasts the LM head to fp32. That is what keeps QLoRA stable in
fp16 — and fp16 was now forced by the T4. Added it, plus warmup, gradient
clipping at 0.3, and `paged_adamw_8bit`; learning rate lowered 2e-4 → 1e-4,
since the widely-quoted 2e-4 assumes bf16's wider dynamic range.

*(A related self-inflicted error: I added `warmup_ratio` without checking it
existed. It does not — the field is `warmup_steps`. Fixed by constructing the
configs locally from the scripts' own kwargs rather than only checking field
names.)*

### Failure 4 — The collapse that was not a collapse

After training, generation degenerated into a single repeated token:

```
{"systemsystemsystemsystemsystem...
```

This looks exactly like a diverged model. **But the loss curve was healthy** —
3.64 → 0.89, no NaN, steady decline.

The real cause: the sanity-check cell called `generate()` immediately after
`train()`, so the model was **still in training mode** — LoRA dropout active,
KV cache disabled. Greedy decoding in that state degenerates regardless of
checkpoint quality.

Adding `model.eval()` produced the correct output from the *same* checkpoint:

```json
{"sensitive": true, "category": "email", "confidence": 1.0}
```

**Lesson: a bad output does not mean a bad model. Check the loss curve before
retraining.**

### Failure 5 — Loss computed over the wrong tokens

The SFT dataset was a single `text` field, so loss covered the entire
sequence. The prompt is a long, varied document excerpt; the answer is a
57-character JSON object. Almost all gradient signal went into learning to
reproduce *document context* rather than the label — which is why loss
plateaued near 0.9 instead of falling.

Fixed by restructuring to `{"prompt", "completion"}`, which TRL recognises as
a prompt-completion dataset and masks the prompt from the loss. Loss then fell
to ~0.0001.

### Failure 6 — A degenerate DPO dataset

The first DPO run looked like a triumph: `rewards/accuracies: 1.0` in 14
steps, `margins: 8.36`, loss 0.367 → 0.0015.

Inspecting the data explained why:

```
DISTINCT chosen completions  : 1   (all 54 identical)
DISTINCT rejected completions: 1   (all 54 identical)
```

Every pair asked the model to prefer the *same string* over the *same other
string*. The optimal policy was "always answer `sensitive: true`" **without
reading the prompt at all.**

Fixed by adding contrastive pairs in the opposite direction — reference-shaped
spans that are *not* people (`"the Borrower"`, `"the Employee"`, `"the Board
of Directors"`) where the correct answer is not-sensitive. 54 → **108 pairs**,
both directions, so the correct answer depends on the input.

**Lesson: a metric that looks too good is a bug report.**

### Failure 7 — Train/inference distribution mismatch

The first honest evaluation gave **precision 0.182** — 279 false positives
across 12 documents. The model flagged 28 of every 38 candidates.

Two distinct causes:

**(a) Class balance.** Trained on a 1:1 positive/negative split, but only ~20%
of candidates at inference are real PII. The model learned a 50% prior for a
20% world.

**(b) Negative distribution — the larger problem.** Training negatives were
arbitrary word n-grams; inference candidates are capitalised entity phrases:

| Trained to reject | Actually asked about |
|---|---|
| `'business focuses on developing'` | `'Suryam Tech Solutions Ltd'` |
| `'regulatory requirements, the prospectus'` | `'Companies Act'` |
| `'to be utilized for'` | `'Gurgaon'`, `'Tech Park'` |

**The model had never once seen a company name labelled "not sensitive."**

Fixed by drawing training negatives from **the same candidate generator used
at inference**, extracted into `app/detection/candidates.py` and imported by
both training and inference so they cannot drift apart again. Negative ratio
set to 3:1. Dataset grew 808 → **1,603 examples**.

Result: precision **0.182 → 0.900**. False positives 279 → 5.

---

## 5. Two evaluation methodology failures

These matter more than any single bug, because they would have invalidated the
results rather than merely degraded them.

### 5.1 — Testing on the training set

The first Phase 3 numbers were measured on documents that were **in** the
training set. `prepare_sft_dataset.py` builds from every file in
`data/synthetic_docs`, and the eval scored files from that same directory:

```
OVERLAP: 12/12 eval documents were also training documents
'Arun Kumar Mishra'      trained_on=True
"the promoter's spouse"  trained_on=True
```

Every Phase 3 number reported before this point measured **memorisation**.

**Fix:** a separate held-out corpus (`data/heldout_docs`) that training never
reads; `--docs-dir`/`--labels-dir` flags; a **warning** printed when a local
adapter is scored on its own training corpus; and
`held_out_from_training: true/false` recorded in every results file so a
leaked run can never be mistaken for a clean one.

**Lesson: the split must be structural, not a convention you remember.**

### 5.2 — Generalising from a biased spot check

Before running the eval, I compared SFT-only against SFT+DPO on six
hand-picked examples. SFT scored 5/6, DPO 3/6, and DPO missed two obvious
names. I concluded DPO had damaged the model and reported that.

**The eval showed the opposite.** DPO halved false positives (279 → 140) with
true positives unchanged (62), and on held-out data it is the *more precise*
model.

Four of my six cases were positives — ~67% — against a real distribution of
about six true spans per 38 candidates, ~16%. DPO's entire purpose was
suppressing over-flagging; measured on a positive-heavy sample, doing its job
correctly looks like damage.

**Lesson: a hand-picked spot check is not an evaluation. If its class balance
does not match deployment, it can invert your conclusion.**

---

## 6. Bugs found in a systematic code review

After the pipeline worked, a deliberate review pass found three defects.

### 6.1 — The local detector redacted only the first occurrence

```python
idx = page_text.find(item["text"])   # returns ONE match
```

A name appearing three times in a contract produced **one** redaction box and
left the other two **legible in the redacted output**. In a tool whose entire
premise is that redaction failures are invisible.

The verifier did not catch it either — the span it was asked to check *had*
been removed.

The API detector already scanned every occurrence, so two implementations of
the same step had silently diverged. Fixed by extracting one shared
`map_results_to_flagged`, with regression tests covering repeated occurrences,
empty needles (which would otherwise loop forever), low-confidence routing,
and hallucinated text.

### 6.2 — The verifier could report PASSED with PII still readable

`verify_redaction` compared flagged strings against re-extracted text using an
**exact substring** test. PDF extraction does not guarantee identical spacing,
so a name returning as `"Rajesh  Mehta"` or split across a line break would
not match — and the verifier would report **passed** with the PII sitting
there readable.

A false negative in the safety net itself: the worst failure this project can
have. Fixed by normalising whitespace on both sides, which errs toward
*detecting* leftovers — a false alarm costs one capped retry, a miss ships
unredacted PII.

### 6.3 — The retry edge was vacuous

Redaction is deterministic. Re-running it with identical spans produced a
byte-identical file that failed verification identically — the retry could
never succeed, it only burned a cycle before hitting the cap.

Fixed by having each retry **widen the redaction boxes**. A box a fraction
tighter than its glyphs is the usual reason text survives `apply_redactions`,
so escalation gives the retry something real to change.

### 6.4 — Silent provider exhaustion (found via a corrupted eval)

A 67-document eval scored recall 0.738. Split by batch:

| | docs 000–039 | docs 040+ |
|---|---|---|
| Recall | 0.991 | **0.432** |
| False positives | 57 | **8** |

Near-zero false positives *with* collapsed recall is the signature of a
detector returning **nothing**, not one making mistakes. Documents and labels
were verified correct — the API calls had failed.

Only two failures were logged, because `_parse_json_with_retry` returned an
empty list on unparseable output **without saying so**, making a failed API
call indistinguishable from a chunk containing no PII.

**Fixed with an ordered provider chain.** Previously: one provider, one
fallback, then give up. Now every configured provider is attempted, each
failure is logged, and if *every* chunk of a document fails,
`detect_sensitive_spans` **raises** rather than returning `[]` — regex-only
output must not be scored as detection output. A generic OpenAI-compatible
adapter means adding a fourth provider is configuration, not code.

*(A recommendation error of mine: I suggested Cerebras as a third provider
from stale knowledge. It returns HTTP 402 — payment required — with no usable
free tier. Verified and documented in `.env.example` rather than left as a
trap.)*

---

## 7. Final results

### 7.1 — Phase 1/2: API detector

Measured on 17 held-out documents (101 ground-truth spans):

| Metric | Value |
|---|---|
| Precision | **0.800** |
| Recall | **0.990** |
| **F1** | **0.885** |
| Verification catch rate | **1.000** |
| False alarm rate | **0.000** |
| Human-review rate | 0.765 |
| Detection latency | 8.3 s/doc |

*Consistent with an earlier 37-document run: P 0.798 / R 0.946 / F1 0.865.*

### 7.2 — Phase 3: API vs fine-tuned local model

All three on the **same 17 held-out documents**, never seen in training.
Regex pass identical; only the context detector swapped.

| | API (Gemini/Groq) | SFT only | SFT + DPO |
|---|---|---|---|
| Precision | 0.800 | 0.814 | **0.900** |
| Recall | **0.990** | 0.475 | 0.446 |
| F1 | **0.885** | 0.600 | 0.596 |
| True / false positives | 100 / 25 | 48 / 11 | 45 / 5 |
| **Missed spans** | **1** | 53 | 56 |
| Latency (uncontended) | **8.3 s/doc** | ~190 s/doc | ~160 s/doc |
| Runs offline | ✗ | ✓ | ✓ |

### 7.3 — The honest conclusion

**The API model wins decisively, and recall is why.** It misses 1 span out of
101; the fine-tuned models miss more than half. For a redaction tool that gap
is disqualifying — a missed identifier is a leak, while a false positive is a
redundant black box. The local model's only real advantage is running offline.

**DPO worked as designed.** It raised precision 0.814 → 0.900 and cut false
positives from 11 to 5. F1 barely moved because the precision gain was paid
for in recall.

**The recall shortfall is attributable, not mysterious.** The candidate
proposer offers **96%** of ground-truth spans to the classifier (97 of 101),
so the ceiling is not the bottleneck — **the classifier rejects about half of
what it is correctly shown.** That traces to one hyperparameter,
`NEGATIVE_RATIO = 3`. At 1:1 the model over-flagged catastrophically
(precision 0.182); 3:1 overshot into under-flagging. **2:1 is the obvious next
experiment.**

### 7.4 — Full result history

| Run | Docs | Held out | P | R | F1 | Note |
|---|---|---|---|---|---|---|
| Phase 1, first eval | 37 | n/a | 0.798 | 0.946 | 0.865 | API baseline |
| Phase 1, 67 docs | 67 | n/a | 0.821 | 0.738 | 0.777 | **degraded** — quota exhaustion |
| API, 12-doc subset | 12 | no | 0.719 | 0.972 | 0.826 | comparison baseline |
| SFT-only, 12-doc | 12 | **no — leaked** | 0.182 | 0.873 | 0.301 | pre-fix, over-flagging |
| SFT+DPO, 12-doc | 12 | **no — leaked** | 0.307 | 0.873 | 0.454 | DPO halved FPs |
| **API, held-out** | 17 | **yes** | **0.800** | **0.990** | **0.885** | final |
| **SFT-only, held-out** | 17 | **yes** | 0.814 | 0.475 | 0.600 | final |
| **SFT+DPO, held-out** | 17 | **yes** | 0.900 | 0.446 | 0.596 | final |

---

## 8. Engineering practices worth noting

- **32 tests**, all passing; CI runs the suite on every push, and the one test
  needing live API keys skips itself when they are absent.
- **Zero lint errors** (`ruff`) across the codebase.
- **Every metric traces to a committed JSON file.** A consistency check
  confirms no figure in any document lacks a backing results file.
- **Failed runs are kept, not deleted.** `results_phase1_67docs_degraded.json`
  is committed with its batch-split analysis, because free-tier quota
  exhaustion mid-evaluation is a real operational finding.
- **Secrets never committed.** `.env` gitignored; a scan confirmed no
  API-key-shaped string in any tracked file before the first public push.
- **Synthetic data only.** No real PII was ever processed — using real
  documents would create the exact privacy problem the tool exists to solve.

---

## 9. Known limitations (stated, not hidden)

1. **Confidence is not calibrated.** Training used exactly `1.0` for clear
   cases and `0.6` for ambiguous ones, so the model reproduces two constants
   rather than estimating uncertainty. The 0.7 threshold routes human review
   correctly — but because of a hardcoded split in the data, not because the
   model measures its own uncertainty.
2. **Long documents are untested.** The eval set is ~300-word excerpts; real
   filings run 50–100+ pages. Chunking exists but end-to-end accuracy at that
   scale is unmeasured.
3. **Partial leftovers pass verification.** The verifier checks whole flagged
   strings; if "Mehta" were removed but "Rajesh" left, it reports passed.
   Token-level checking would fix this.
4. **`per_doc_detection` stores counts, not span texts.** Precision cannot yet
   be broken down by category.
5. **Small held-out set.** 17 documents, 101 spans — enough for a clear
   signal, not enough for tight confidence intervals.
6. **"Runs offline" applies only to the Phase 3 configuration.** Phase 1/2
   send document text to Gemini/Groq.

---

## 10. Claims that must NOT be made

- ❌ "Documents never leave your machine" — true only of Phase 3.
- ❌ "Fine-tuning improved PII detection" — it did **not** beat the API model
  (F1 0.596 vs 0.885) and is ~20× slower on CPU. It is more *precise* (0.900
  vs 0.800) but misses more than half the PII.
- ❌ Any accuracy figure not present in `eval/results_*.json`.
- ❌ "99% accuracy" — never measured, and accuracy is the wrong metric for
  span detection.

---

## 11. Summary of quantified achievements

| Achievement | Number |
|---|---|
| Detection F1 on held-out data | **0.885** |
| Recall on held-out data | **0.990** (1 miss in 101 spans) |
| Verification catch rate | **1.000**, 0 false alarms |
| False positives eliminated by fixing distribution mismatch | **279 → 5** |
| Precision improvement from that fix | **0.182 → 0.900** |
| Local-model precision vs Gemini | **0.900 vs 0.800** |
| Training examples generated | 1,603 SFT + 108 DPO preference pairs |
| Labelled evaluation documents | 67 training + 17 held-out |
| Tests | 32 passing, CI on every push |
| Distinct bugs found and fixed | 10+ |
| Latency, API path | 8.3 s/doc |

---

## 12. What this project demonstrates

**Systems engineering.** A six-stage pipeline with swappable components behind
stable interfaces — swapping the API detector for a fine-tuned local model is
one import line, because the contract was designed first.

**ML engineering.** QLoRA fine-tuning (SFT + DPO) on constrained free-tier
hardware, including hardware-specific numerical issues (bf16 unavailable on
Turing), library API migration, and diagnosing a train/inference distribution
mismatch from its symptom.

**Evaluation rigour.** Ground truth generated at creation time rather than
re-detected; generation and detection deliberately use different model
families; a deliberately-broken-input test for the verifier; and a held-out
split built *after* discovering the eval was scoring on training data.

**Engineering judgement.** Reporting that the fine-tuned model lost, and why.
Keeping a failed eval run as evidence. Correcting a published conclusion when
better data contradicted it. Attributing a recall shortfall to a single named
hyperparameter instead of shrugging at it.

**The debugging that matters most** was rarely the obvious kind. The model
that looked collapsed was fine — the test harness was wrong. The DPO run that
looked perfect was degenerate. The eval that looked like a model failure was
exhausted API quota. The redaction tool had a bug that left PII visible, in
exactly the way the project exists to prevent.
