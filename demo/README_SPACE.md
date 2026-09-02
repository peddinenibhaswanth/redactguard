---
title: RedactGuard
emoji: 🛡️
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: "1.63.0"
app_file: demo/app.py
pinned: false
---

# RedactGuard demo

Live demo of the RedactGuard PII redaction pipeline. See the main repo README
for the full project writeup, architecture, and eval numbers.

Deployment preserves the repo's folder structure (`app/` and `demo/` as
siblings) rather than flattening `app.py` to the Space root - `demo/app.py`
locates the `app` package relative to its own folder, so moving it out of
`demo/` breaks the import. `app_file: demo/app.py` above tells the Space
where to find it without needing to flatten anything.

See the main README's "Deploying the demo" section for the full copy-paste
walkthrough.
