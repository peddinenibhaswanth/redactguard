"""CLI entrypoint. Faster to debug against than the API during Phase 1/2 -
`app/api.py` wraps the same graph once this is stable.

Phase 1 acceptance test: run this against a real (test) PDF with fake PII in
it, get a redacted output file, open it in a PDF viewer, and confirm - by
selecting text in the redacted areas - that nothing is copyable.
"""
import argparse
import os

from app.config import OUTPUT_DIR
from app.graph.pipeline_graph import build_graph
from app.report.report_generator import save_report


def run_pipeline(file_path: str) -> dict:
    graph = build_graph()
    final_state = graph.invoke({"input_path": file_path})

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    save_report(final_state["report"], OUTPUT_DIR, base_name)

    return final_state


def main():
    parser = argparse.ArgumentParser(description="RedactGuard - PII detection, redaction, and verification.")
    parser.add_argument("--file", required=True, help="Path to a .pdf or .docx file to redact.")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        raise SystemExit(f"File not found: {args.file}")

    final_state = run_pipeline(args.file)
    report = final_state["report"]

    print(report["markdown_summary"])
    print(f"\nRedacted file: {report['output_path']}")
    print(f"Full report: {OUTPUT_DIR}/{os.path.splitext(os.path.basename(args.file))[0]}_report.json")


if __name__ == "__main__":
    main()
