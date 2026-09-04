# RedactGuard — project status

Living checklist. Updated as work completes so both of us can see what is
done, what is running, and what is left.

**Last updated:** 2026-09-04 — Phase 3 complete, all evals done.

---

## ✅ Shipped and verified

| Item | Evidence |
|---|---|
| Phase 1 pipeline (extract → regex + LLM detect → redact → verify → report) | `app/`, runs end-to-end on `samples/` |
| LangGraph orchestration with capped redact/verify retry loop | `app/graph/pipeline_graph.py` |
| Phase 2 eval harness (precision/recall/F1, verification catch rate, human-review rate) | `eval/run_eval.py` |
| **Live demo, public** | https://redactguard-66iybemnkbkxq6ngdvxqkx.streamlit.app/ |
| **GitHub repo, public** | https://github.com/peddinenibhaswanth/redactguard |
| FastAPI service + upload validation | `app/api.py`, `tests/test_api_validation.py` |
| Dockerfile + GitHub Actions CI | `Dockerfile`, `.github/workflows/ci.yml` |
| Test suite — **32 passing** | `tests/` |
| Synthetic sample document for the demo | `samples/sample_loan_agreement.pdf` |
| Phase 3 SFT + DPO training on free-tier T4 | `phase3_finetune/`, both adapters exported |
| Held-out test split (17 docs, 101 spans) | `data/heldout_docs/`, never used in training |
| Interview prep + resume bullets | `docs/INTERVIEW_PREP.md` |

### Final measured results — 17 held-out documents, 101 spans

All three configurations on the **same documents**, which the fine-tuned
models were never trained on. Regex pass identical; only the context
detector swapped.

| | API (Gemini/Groq) | SFT only | SFT + DPO |
|---|---|---|---|
| Precision | 0.800 | 0.814 | **0.900** |
| Recall | **0.990** | 0.475 | 0.446 |
| F1 | **0.885** | 0.600 | 0.596 |
| Missed spans | **1** | 53 | 56 |
| Runs offline | ✗ | ✓ | ✓ |

Verification catch rate **1.000**, false alarm rate **0.000**.

**Conclusion:** the API model wins decisively, and recall is why — it misses
1 span in 101 where the local models miss more than half. For a redaction
tool that is disqualifying. The local model's advantage is offline operation
only. The recall shortfall is attributable: the candidate proposer reaches
96% of ground-truth spans, so the classifier is rejecting about half of what
it is correctly shown — a consequence of `NEGATIVE_RATIO = 3`.

---

## 📋 Remaining

| # | Task | Owner |
|---|---|---|
| 1 | Paste the resume bullets into your resume | **you** |

Everything else is done.

### Optional, not blocking

- **Retune `NEGATIVE_RATIO` 3 → 2** in `prepare_sft_dataset.py`. 1:1 gave
  precision 0.182; 3:1 overshot to recall 0.446. Needs a ~45 min Colab
  retrain. Would improve the Phase 3 numbers but not change the conclusion.
- Add a third provider (OpenRouter free tier) — `.env.example` documents how;
  Cerebras was checked on 2026-09-03 and returns HTTP 402 without billing.
- Log actual false-positive/negative span texts per document, not just counts.
- Test on genuinely long documents; the eval set is ~300-word excerpts.

---

## 🐛 Bugs found and fixed during review

Worth knowing — these are interview material, not just changelog entries.

1. **Local detector redacted only the first occurrence of each span.**
   `str.find` returns one match, so a name appearing three times in a contract
   left two legible in the "redacted" output. The API detector already
   scanned every occurrence, so the two had diverged. Extracted the shared
   scan into `map_results_to_flagged`. Regression-tested.

2. **Verifier could report PASSED with PII still readable.** Exact substring
   matching against re-extracted text meant whitespace artefacts
   (`"Rajesh  Mehta"`, or a line break) hid a leftover — a false negative in
   the safety net itself. Now normalises whitespace, which errs toward
   detecting leftovers.

3. **The retry edge was vacuous.** Redaction is deterministic, so retrying
   with identical spans produced an identical file and failed identically.
   Each retry now widens the redaction boxes, addressing the usual cause of a
   surviving glyph.

4. **Silent provider exhaustion.** Both free tiers hitting daily limits made
   detection return empty, indistinguishable from "no PII found" — that is
   how a 67-document eval came to report recall 0.43 and look like a model
   problem. Now an ordered provider chain, and total exhaustion raises.

5. **Train/test leak.** The Phase 3 eval scored on documents that were in the
   SFT/DPO training set. Fixed with a separate held-out corpus plus a warning
   when a local adapter is scored on its own training data.

---

## ⚠️ Claims to avoid on the resume

- ❌ "Documents never leave your machine" — true only of the Phase 3 local
  configuration; Phase 1/2 call Gemini/Groq.
- ❌ "Fine-tuned model improved detection" — it does **not** beat the API
  model. Its advantage is offline operation.
- ❌ Any accuracy figure not in `eval/results_*.json`.
