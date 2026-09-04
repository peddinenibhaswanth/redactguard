# RedactGuard — project status

Living checklist. Updated as work completes so both of us can see what is
done, what is running, and what is left.

**Last updated:** 2026-09-04

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

### Measured results (clean, held-out where it matters)

**API detector (Gemini/Groq) on 17 held-out documents:**
precision **0.800** · recall **0.990** · F1 **0.885** · 8.3 s/doc
verification catch rate **1.0** · false alarm rate **0.0**

**Phase 1 on the original 37-document set:** F1 **0.865** (`eval/results_phase1.json`)

---

## 🔄 Running now

- **SFT+DPO** local adapter on the 17 held-out documents (~55 min)
- **SFT-only** local adapter on the same set (~55 min, queued after)

These are CPU-bound and need no API quota. They produce the only Phase 3
numbers that measure generalisation rather than memorisation.

---

## 📋 Remaining

| # | Task | Owner | Est. |
|---|---|---|---|
| 1 | Write the three-way held-out comparison into the README | me | 15 min |
| 2 | Update `INTERVIEW_PREP.md` bullets with final held-out numbers | me | 10 min |
| 3 | Final consistency pass — every number in every doc traces to a results file | me | 15 min |
| 4 | Hand over final resume bullets | me | — |
| 5 | Paste bullets into your actual resume | **you** | — |

### Optional, not blocking

- Add a third provider (OpenRouter free tier) — `.env.example` documents it;
  Cerebras was checked and needs billing
- Retune `NEGATIVE_RATIO` in `prepare_sft_dataset.py` — the 3:1 ratio pushed
  precision to 0.938 but halved recall; 2:1 may balance better (needs a
  ~45 min Colab retrain)
- Log actual false-positive/negative span texts per document, not just counts

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
