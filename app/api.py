"""FastAPI wrapper around the same graph the CLI uses. Added once the CLI
pipeline was stable, per the guide's own sequencing (CLI first, API once
proven).

Validates uploads before they touch the pipeline: extension AND magic-byte
sniffing (not just trusting the client-supplied extension/content-type),
plus a size cap - a project about handling sensitive documents safely should
not itself accept arbitrary unchecked file uploads.
"""
import os
import uuid

import filetype
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import OUTPUT_DIR
from app.main import run_pipeline

app = FastAPI(title="RedactGuard", description="PII detection, redaction, and verification for PDF/DOCX.")

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20MB
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
ALLOWED_MIME_KINDS = {
    ".pdf": {"pdf"},
    ".docx": {"docx", "zip"},  # docx files are zip containers; filetype often reports "zip"
}
UPLOAD_DIR = os.path.join(OUTPUT_DIR, "uploads")


def _validate_upload(filename: str, content: bytes) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file extension '{ext}'. Only .pdf and .docx are accepted.")

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"File too large ({len(content)} bytes). Max is {MAX_UPLOAD_BYTES} bytes.")
    if len(content) == 0:
        raise HTTPException(400, "Empty file.")

    kind = filetype.guess(content)
    detected = kind.extension if kind else None
    if detected not in ALLOWED_MIME_KINDS[ext]:
        raise HTTPException(
            400,
            f"File content doesn't match its extension (claimed {ext}, detected {detected}). Refusing to process.",
        )
    return ext


@app.post("/redact")
async def redact_endpoint(file: UploadFile = File(...)):
    content = await file.read()
    ext = _validate_upload(file.filename, content)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}{ext}"
    saved_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(saved_path, "wb") as f:
        f.write(content)

    final_state = run_pipeline(saved_path)
    report = final_state["report"]

    return {
        "report": report,
        "download_url": f"/download/{os.path.basename(report['output_path'])}",
    }


@app.get("/download/{filename}")
async def download_endpoint(filename: str):
    safe_name = os.path.basename(filename)  # strip any path traversal attempt
    path = os.path.join(OUTPUT_DIR, safe_name)
    if not os.path.isfile(path):
        raise HTTPException(404, "File not found.")
    return FileResponse(path, media_type="application/pdf", filename=safe_name)


@app.get("/health")
async def health():
    return {"status": "ok"}
