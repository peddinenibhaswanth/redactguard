---
title: RedactGuard
emoji: 🛡️
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: "1.63.0"
app_file: app.py
pinned: false
---

# RedactGuard demo

Live demo of the RedactGuard PII redaction pipeline. See the main repo README
for the full project writeup, architecture, and eval numbers.

## Deploying this yourself (manual - do this from your own machine/account)

Hugging Face Spaces deploys are a git push to a Space-owned remote, so this
needs your HF login, not something that can be done on your behalf here.

1. Create a free account at huggingface.co if you don't have one.
2. Create a new Space: huggingface.co/new-space -> SDK: **Streamlit** -> name
   it (e.g. `redactguard`).
3. On your machine, from the repo root:
   ```bash
   pip install -U huggingface_hub
   huggingface-cli login   # paste an access token from huggingface.co/settings/tokens
   ```
4. Add the Space as a git remote and push this folder's contents to it as the
   Space's root (the Space needs `app.py`, this `README_SPACE.md` renamed to
   `README.md`, and a `requirements.txt` at ITS root - not nested under `demo/`):
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/redactguard space-repo
   cp demo/app.py space-repo/app.py
   cp demo/README_SPACE.md space-repo/README.md
   cp requirements.txt space-repo/requirements.txt
   cp -r app space-repo/app
   cp -r phase3_finetune/prompt_template.py space-repo/phase3_finetune/prompt_template.py
   cd space-repo && git add -A && git commit -m "Deploy RedactGuard demo" && git push
   ```
5. In the Space's Settings -> Repository secrets, add `GEMINI_API_KEY` and
   `GROQ_API_KEY` (same values as your local `.env` - Spaces secrets are
   private to you, never exposed to visitors).
6. The Space will build and give you a public URL
   (`https://huggingface.co/spaces/<your-username>/redactguard`) - that's the
   link for your resume/portfolio.

Note: the free CPU tier is enough for this demo (Phase 1/2 detection calls
Gemini/Groq APIs, no local inference). If you later demo Phase 3's local
model on the Space, CPU inference of the 0.5B adapter is still fine on free
tier, just slower per request.
