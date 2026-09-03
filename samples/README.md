# Sample document

`sample_loan_agreement.pdf` — a synthetic loan-agreement extract for trying the
live demo or the CLI. **Every name, identifier, and address in it is invented**
(emails use the reserved `example.com` domain, the PAN is the standard dummy
pattern, the Aadhaar-shaped number is a repeating-digit placeholder). It
describes no real person or company.

It's deliberately built so a single upload exercises the whole pipeline:

| Detector | What it should catch |
|---|---|
| regex | email, Indian phone, PAN, Aadhaar-shaped number, IFSC code |
| LLM | two full names, a street address, one indirect reference |
| neither | ordinary contract prose — so over-flagging is visible |

Expected result (measured, Phase 1 config): **9 spans redacted** (5 regex + 4
LLM), verification passes on the first attempt with 0 retries, and exactly one
span — *"the promoter's spouse"* at confidence 0.60 — is flagged for human
review rather than trusted silently.

Try it:

```bash
python -m app.main --file samples/sample_loan_agreement.pdf
```

Then open the redacted output in a PDF viewer and try to select the blacked-out
text. Nothing should be copyable — that's the difference between real redaction
and a drawn rectangle.

Regenerate the PDF with `python samples/generate_sample.py`.
