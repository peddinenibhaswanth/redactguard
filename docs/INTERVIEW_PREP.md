# RedactGuard — resume bullets and interview prep

Numbers in this file are measured, not estimated. Every figure traces to a
committed artifact (`eval/results_*.json`) so you can show the evidence if
asked. **Do not round them up.** Being able to say "0.798, and here is the
file" is worth more than a rounder number you cannot defend.

---

## Resume bullets

Pick 3-4. The first two carry the most weight; the third is the differentiator
most candidates cannot claim.

- Built a document PII-redaction pipeline (LangGraph, PyMuPDF, Gemini/Groq)
  combining regex detection for structured identifiers with an LLM pass for
  context-dependent spans (names, addresses, indirect references), reaching
  **F1 0.865 / recall 0.946** on a 37-document labelled evaluation set.

- Designed a **redaction verification stage** that re-extracts the output PDF
  and confirms flagged text is unrecoverable, routing failures back through
  redaction as an explicit graph edge with a retry cap; validated by injecting
  deliberately-broken (overlay-only) redactions into ~15% of eval runs and
  measuring a **100% catch rate with 0 false alarms**.

- Fine-tuned Qwen2.5-0.5B with **QLoRA (SFT + DPO)** on a free-tier T4 to
  replace the API detector with an offline local model behind an identical
  interface; DPO **halved false positives (279 → 140) at zero recall cost**,
  lifting F1 from 0.301 to 0.454, and the API-vs-local comparison is reported
  with its real tradeoffs rather than only the favourable numbers.

- Shipped with a **live Streamlit demo**, FastAPI service with upload
  validation, Dockerfile, pytest suite and GitHub Actions CI.

### Bullets to avoid

- ❌ "Achieved 99% accuracy" — not measured, and accuracy is the wrong metric
  for span detection.
- ❌ "Ensures documents never leave your machine" — only true of the Phase 3
  local-model configuration. Phase 1/2 send text to Gemini/Groq. Scope the
  claim or drop it.
- ❌ "Fine-tuned an LLM to improve PII detection" — the fine-tuned model does
  **not** beat the API model (F1 0.454 vs 0.826) and is ~30x slower on CPU.
  DPO improved the local model over SFT alone; it did not close the gap to
  Gemini. Its one real advantage is running offline.

---

## The three questions you will definitely be asked

### 1. "Why LangGraph? Couldn't this be a plain script?"

Yes, and honestly a `while` loop would work. The reason it is a graph is that
verification can send a document *back* to redaction, and expressing that as a
conditional edge with an explicit retry cap makes the control flow inspectable
— you can see the state at each node, log the retry count, and explain the
loop without reading nested loop logic. It also made swapping the detection
node for the fine-tuned model a one-line change.

Do not oversell it. The honest framing is "chosen for explicit, debuggable
control flow", not "needed for scale".

### 2. "What is actually novel here? Redaction tools exist."

Most redaction code draws a black rectangle. `page.draw_rect()` and
`page.add_redact_annot()` + `page.apply_redactions()` produce files that look
identical, but only the second removes the underlying text object — the first
leaves everything selectable and copyable. That failure is invisible to the
eye, which is exactly why it ships.

So the pipeline treats its own redaction as unverified: it re-extracts the
output file and checks the flagged strings are actually gone. To prove the
verifier works I inject that exact bug on purpose in ~15% of eval runs and
measure how many it catches.

**Have the demo open.** Redact the sample document, then try to select the
blacked-out text. Nothing is copyable. That gesture lands better than the
explanation.

### 3. "Walk me through the fine-tuning."

Phase 1 uses Gemini for context-dependent detection. Phase 3 replaces that one
node with a locally fine-tuned Qwen2.5-0.5B — same interface, so it is a
one-import swap — trained with QLoRA on Colab's free T4: SFT first to teach
the task, then DPO to bias toward flagging ambiguous references.

The interesting part is that **DPO made it worse**, and I can explain why (see
below). SFT worked; DPO regressed name detection. I reported both.

---

## The debugging stories worth telling

These matter more than the metrics. They are the evidence you actually ran
this rather than following a notebook.

### bf16 on a T4

The training scripts hardcoded `bf16=True`, but a T4 is Turing (sm_75) and has
no bfloat16 support at all — bf16 needs Ampere. Detect at runtime with
`torch.cuda.is_bf16_supported()` and fall back to fp16, for both the trainer
and the 4-bit compute dtype.

### The collapse that was not a collapse

After SFT, generation degenerated into one repeated token — the classic look
of a diverged model. But the loss curve was healthy (3.64 → 0.89, no NaN).
The real cause: the sanity-check cell called `generate()` immediately after
`train()`, so the model was still in training mode with LoRA dropout active
and the KV cache disabled. `model.eval()` fixed it; the checkpoint had been
fine all along.

**Lesson to state:** a bad output does not mean a bad model. Check the loss
curve before you retrain.

### Completion-only loss

The SFT dataset was a single `text` field, so loss covered the whole sequence
— the model spent most of its capacity learning to reproduce long document
excerpts rather than the short JSON label. Restructuring to
`{"prompt", "completion"}` lets TRL mask the prompt so gradients come only
from the answer.

### The degenerate DPO dataset

First DPO run hit `rewards/accuracies: 1.0` in 14 steps with `margins: 8.36`.
That looked like success. It was not: every one of the 54 pairs had an
*identical* chosen string and an *identical* rejected string, so the optimal
policy was "always answer sensitive: true" without reading the prompt at all.

Fix: add contrastive pairs in the opposite direction — reference-shaped spans
that are *not* people ("the Borrower", "the Employee", "the Board of
Directors") where the correct answer is not-sensitive. That forces the
decision to depend on the input.

**Lesson to state:** a metric that looks too good is a bug report. Check
whether your task is actually learnable-by-shortcut before believing the
number.

### Judging a model on a biased spot check (the mistake worth admitting)

Before running the eval I sanity-checked SFT-only against SFT+DPO on six
hand-picked examples. SFT scored 5/6, DPO 3/6, and DPO missed two obvious
names. The obvious read was that DPO had damaged the model.

The eval said the opposite. On 12 documents DPO **halved false positives
(279 → 140) with true positives unchanged (62)**, taking precision from 0.182
to 0.307 and F1 from 0.301 to 0.454.

The spot check was misleading because four of its six cases were positives —
roughly 67% — whereas the real distribution is about six true spans among 38
candidate spans, roughly 16%. DPO's entire purpose was to suppress
over-flagging. Measured on a positive-heavy sample, doing its job correctly
looks like damage.

**Lesson to state:** a hand-picked spot check is not an evaluation. If the
class balance of your sample does not match deployment, it can invert your
conclusion — and it did here. This is also why the eval set is generated with
ground truth attached rather than assembled by hand.

If asked "did anything surprise you", this is the answer to give.

### QLoRA adapters across quantization levels

The adapter was trained against a 4-bit NF4 base on Colab but runs against a
full-precision base locally, and the predictions do not match — the same input
that returned `sensitive: true` in Colab returns `false` locally. Adapters
trained on a quantized base do not transfer cleanly to an unquantized one.
Worth knowing before you promise "train in Colab, deploy anywhere".

---

## Questions where honesty beats polish

**"Is your confidence score calibrated?"**
No. Training examples used exactly `1.0` for clear-cut cases and `0.6` for
ambiguous ones, so the model reproduces two constants rather than estimating
uncertainty. The `0.7` threshold separates them correctly, so human-review
routing works — but it works because of a hardcoded split in the data, not
because the model measures its own uncertainty.

**"Why is precision only ~0.80?"**
The LLM over-flags: capitalised entities that are organisations rather than
people get caught. That is the intended direction — the system prompt
explicitly says over-flagging beats missing PII, and low-confidence spans are
surfaced for human review rather than silently trusted. But the eval currently
records per-document tp/fp/fn counts without the individual mismatched spans,
so I cannot break the false positives down by category. That is the next thing
I would add.

**"How does it handle a 100-page filing?"**
It chunks per page for the LLM call, but the eval set is ~300-word excerpts,
so end-to-end accuracy on genuinely long documents is not measured. I would
not claim it works at that scale without testing it.

**"Why synthetic data?"**
Real financial and legal documents with known ground-truth PII are not
available, and using real ones creates the privacy problem the tool exists to
solve. Ground truth is emitted at generation time rather than re-detected
afterwards — re-detecting would just be testing the detector against itself.
Generation uses a different model family than detection for the same reason.

**"What would you do with another week?"**
Log the actual false-positive and false-negative spans per document so
precision can be broken down by category; test on genuinely long documents;
and rebalance the SFT set — indirect references are only ~7% of training
examples, which is the specific weakness DPO was meant to patch and could
probably be fixed directly in SFT instead.
