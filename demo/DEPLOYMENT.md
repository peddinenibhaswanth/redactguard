# Deploying the RedactGuard demo

## Recommended: Streamlit Community Cloud (free)

Purpose-built for Streamlit apps, free, and supports private secrets. Needs
the repo on GitHub first.

1. Push this repo to GitHub (see the main README).
2. Go to **share.streamlit.io** and sign in with GitHub.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Fill in:
   - **Repository:** `<your-username>/redactguard`
   - **Branch:** `main`
   - **Main file path:** `demo/app.py`
5. Before clicking Deploy, open **Advanced settings…** → **Secrets**, and paste
   (TOML format, quotes required):
   ```toml
   GEMINI_API_KEY = "your-real-key"
   GROQ_API_KEY = "your-real-key"
   ```
6. Click **Deploy**. First build takes a few minutes.

`demo/app.py` bridges `st.secrets` into environment variables before importing
`app.config`, so the same code runs unchanged on Streamlit Cloud, in Docker,
and locally with a `.env`.

### Note on OCR and DOCX support

Streamlit Cloud installs apt packages from a `packages.txt` at the **repo
root**. This repo keeps one at `demo/packages.txt` (listing `tesseract-ocr`
and `libreoffice`). If you want scanned-PDF or DOCX support on the deployed
demo, copy it to the repo root and push:

```bash
cp demo/packages.txt packages.txt
```

Without it the demo still works fine for digital-text PDFs, which is the
common case for a portfolio demo. Note that `libreoffice` is large and slows
the build considerably - if you only care about PDFs, drop that line.

## Hugging Face Spaces (requires a PRO subscription)

As of 2026, Hugging Face only offers **Static** Spaces on the free tier;
Gradio and Docker Spaces require PRO, and Streamlit is no longer offered as a
first-class SDK at all. A Space created with `sdk: streamlit` metadata will be
accepted but sits **Paused** and never runs without a paid plan.

If you do have PRO, the practical route is a **Docker** Space using the
repository's root `Dockerfile` (which already installs `tesseract-ocr` and
`libreoffice`), with `GEMINI_API_KEY` / `GROQ_API_KEY` set under
Settings → Variables and secrets.

## Other free options

- **Render** free web service - works, but the instance spins down after
  inactivity, so the first visit after an idle period takes ~30s to wake.
- **Railway / Fly.io** free allowances - both fine; Fly needs the `Dockerfile`.

## Self-hosting with Docker

```bash
docker compose up --build      # serves the FastAPI API on :8000
```

The compose service runs `app/api.py`, not the Streamlit demo - the API is the
deployment target for programmatic use, the Streamlit app for the human-facing
demo.
