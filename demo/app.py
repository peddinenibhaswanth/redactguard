"""Streamlit demo UI. Not part of the original phased build (CLI -> API) -
added because a clickable live demo is worth more to a reviewer than
instructions to clone and run a CLI command.

Runs the same graph the CLI/API use (app.main.run_pipeline), so there is
exactly one pipeline implementation, not a UI-specific reimplementation.

Cleans up the redacted output + report files from disk right after reading
them into memory for the download button - this runs as a shared public
demo on Hugging Face Spaces, and a PII-redaction tool leaving PII-adjacent
report content sitting in a shared outputs/ directory would be a genuinely
embarrassing thing for this specific project to get wrong.
"""
import os
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Streamlit Community Cloud supplies secrets through st.secrets, while
# app/config.py reads plain environment variables - and it reads them at
# import time, into module-level constants. So the bridge has to happen
# BEFORE app.config is imported below, or the keys read as empty strings and
# detection silently degrades to regex-only. Hosts that already provide real
# env vars (Docker, a local .env) are left untouched.
for _key in ("GEMINI_API_KEY", "GROQ_API_KEY"):
    if not os.environ.get(_key):
        try:
            if _key in st.secrets:
                os.environ[_key] = str(st.secrets[_key])
        except Exception:
            pass  # no secrets.toml on this host - env vars are the source instead

from app.config import OUTPUT_DIR  # noqa: E402
from app.main import run_pipeline  # noqa: E402

st.set_page_config(page_title="RedactGuard", page_icon="🛡️", layout="wide")

st.title("🛡️ RedactGuard")
st.caption("PII detection, true redaction, and automatic verification for PDF/DOCX documents.")

with st.expander("How this works", expanded=False):
    st.markdown(
        """
1. **Extract** text + coordinates from the uploaded PDF/DOCX.
2. **Detect** PII with regex (structured IDs: emails, PAN, phone, etc.) plus an
   LLM pass for context-dependent spans (names, indirect references, addresses).
3. **Redact** using PyMuPDF's true redaction API - this strips the underlying
   text object, not just draws a black box over it.
4. **Verify** by re-extracting the redacted file and confirming nothing
   flagged survived. If something did, redaction is retried automatically
   (capped retries) before the document is flagged for manual handling.

This demo configuration calls Gemini/Groq for step 2, so uploaded documents
are sent to those APIs for detection. The project's Phase 3 (see README) swaps
in a fine-tuned local model so this step runs fully offline.
        """
    )

uploaded = st.file_uploader("Upload a PDF or DOCX", type=["pdf", "docx"])

if uploaded is not None:
    if st.button("Run redaction pipeline", type="primary"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, uploaded.name)
            with open(input_path, "wb") as f:
                f.write(uploaded.getbuffer())

            with st.spinner("Extracting, detecting, redacting, verifying..."):
                try:
                    final_state = run_pipeline(input_path)
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")
                    st.stop()

            report = final_state["report"]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Spans redacted", report["total_spans_redacted"])
            col2.metric("Verification", "PASSED" if report["verification_passed"] else "FAILED")
            col3.metric("Retries needed", report["redact_verify_retries"])
            col4.metric("Needs human review", len(report["spans_needing_human_review"]))

            st.subheader("By category")
            st.json(report["spans_by_category"])

            if report["spans_needing_human_review"]:
                st.subheader("Flagged for human review (low confidence)")
                st.table(report["spans_needing_human_review"])

            with open(report["output_path"], "rb") as f:
                redacted_bytes = f.read()

            base_name = os.path.splitext(os.path.basename(input_path))[0]
            for path in [report["output_path"]] + [
                os.path.join(OUTPUT_DIR, f"{base_name}_report.json"),
                os.path.join(OUTPUT_DIR, f"{base_name}_report.md"),
            ]:
                try:
                    os.remove(path)
                except OSError:
                    pass

            st.subheader("Download")
            st.download_button(
                "Download redacted PDF",
                redacted_bytes,
                file_name=f"{base_name}_redacted.pdf",
                mime="application/pdf",
            )

            with st.expander("Full JSON report"):
                st.json(report)
else:
    st.info("Upload a PDF or DOCX to get started. Nothing is stored beyond this session.")

st.divider()
st.caption(
    "RedactGuard - PII detection, redaction, and verification pipeline "
    "(LangGraph orchestration, regex + LLM detection, PyMuPDF true redaction)."
)
