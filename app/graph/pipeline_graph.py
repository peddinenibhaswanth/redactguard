"""LangGraph wiring. The verification step needs to be able to send the
document back to redaction if leftover text is found - LangGraph is used
(instead of a plain while-loop script) because it makes that retry/loop-back
explicit as a graph edge with a max-retry guard, which is easier to reason
about, log, and explain than nested loop logic.

retry_count is incremented inside redact_node itself so the conditional edge
always sees accurate state when it decides report-vs-redact-again.
"""
import os
from typing import List, TypedDict

from langgraph.graph import END, StateGraph

from app.config import MAX_REDACT_RETRIES, OUTPUT_DIR
from app.detection.base import FlaggedSpan
from app.detection.llm_detector import detect_llm_spans
from app.detection.regex_detector import detect_regex_spans
from app.extraction.base import ExtractedDocument
from app.extraction.docx_extractor import convert_docx_to_pdf
from app.extraction.pdf_extractor import extract_pdf
from app.redaction.redactor import redact_pdf
from app.report.report_generator import generate_report
from app.verification.verifier import verify_redaction


class PipelineState(TypedDict):
    input_path: str
    working_pdf_path: str
    file_type: str
    extracted_doc: ExtractedDocument
    regex_spans: List[FlaggedSpan]
    llm_spans: List[FlaggedSpan]
    flagged_spans: List[FlaggedSpan]
    output_path: str
    verification_result: dict
    retry_count: int
    report: dict


def extract_node(state: PipelineState) -> dict:
    input_path = state["input_path"]
    ext = os.path.splitext(input_path)[1].lower()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    if ext == ".docx":
        working_pdf_path = os.path.join(OUTPUT_DIR, f"{base_name}_converted.pdf")
        convert_docx_to_pdf(input_path, working_pdf_path)
        file_type = "docx"
    elif ext == ".pdf":
        working_pdf_path = input_path
        file_type = "pdf"
    else:
        raise ValueError(f"Unsupported file type: {ext}. Only .pdf and .docx are supported.")

    extracted_doc = extract_pdf(working_pdf_path)
    return {
        "working_pdf_path": working_pdf_path,
        "file_type": file_type,
        "extracted_doc": extracted_doc,
        "retry_count": 0,
    }


def regex_detect_node(state: PipelineState) -> dict:
    spans = detect_regex_spans(state["extracted_doc"])
    return {"regex_spans": spans}


def llm_detect_node(state: PipelineState) -> dict:
    llm_spans = detect_llm_spans(state["extracted_doc"], already_found=state["regex_spans"])
    flagged = state["regex_spans"] + llm_spans
    return {"llm_spans": llm_spans, "flagged_spans": flagged}


def redact_node(state: PipelineState) -> dict:
    base_name = os.path.splitext(os.path.basename(state["input_path"]))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_redacted.pdf")

    redact_pdf(state["working_pdf_path"], output_path, state["flagged_spans"])

    return {"output_path": output_path, "retry_count": state["retry_count"] + 1}


def verify_node(state: PipelineState) -> dict:
    originally_redacted_texts = [s.text for s in state["flagged_spans"]]
    result = verify_redaction(state["output_path"], originally_redacted_texts)
    return {"verification_result": result}


def report_node(state: PipelineState) -> dict:
    report = generate_report(state)
    return {"report": report}


def _route_after_verify(state: PipelineState) -> str:
    verified_ok = state["verification_result"]["passed"]
    retries_exhausted = state["retry_count"] >= MAX_REDACT_RETRIES
    return "report" if (verified_ok or retries_exhausted) else "redact"


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
        _route_after_verify,
        {"report": "report", "redact": "redact"},
    )
    graph.add_edge("report", END)
    return graph.compile()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m app.graph.pipeline_graph <path-to-pdf-or-docx>")
        sys.exit(1)

    app_graph = build_graph()
    final_state = app_graph.invoke({"input_path": sys.argv[1]})
    print(final_state["report"])
