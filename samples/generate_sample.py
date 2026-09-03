"""Generates a sample document for trying the demo.

Every identifier below is invented. Emails use example.com (RFC 2606 reserved
for documentation), the PAN is the standard all-dummy test pattern, and the
Aadhaar-shaped number is a repeating-digit placeholder that cannot be a real
one. Nothing here belongs to a real person or company.

The document deliberately mixes categories so one upload exercises the whole
pipeline:
  - regex detector  -> email, phone, PAN, Aadhaar-shaped, IFSC
  - LLM detector    -> names, street address, indirect reference
  - neither         -> ordinary contract prose, so false positives are visible

Regenerate with:  python samples/generate_sample.py
"""
import os

import pymupdf

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_loan_agreement.pdf")

TITLE = "LOAN AGREEMENT - SCHEDULE II (EXTRACT)"

# Plain ASCII only: PyMuPDF's base-14 "helv" font uses WinAnsi encoding and
# silently drops "smart" quotes/dashes, which would leave gaps in the text layer.
BODY = """This Agreement is executed between Meridian Capital Finance Limited (the "Lender") and the borrower named below. This extract is provided for record purposes and forms part of the principal agreement dated 14 March 2025.

1. BORROWER PARTICULARS

The borrower, Rajesh Anand Mehta, resident of Flat 402, Sunrise Residency, 27 Nehru Road, Bengaluru 560038, Karnataka, has applied for a term loan facility. The borrower may be contacted at rajesh.mehta@example.com or on 9845012376 for all matters relating to servicing of this facility.

For verification purposes the borrower has furnished Permanent Account Number ABCDE1234F and identity reference 9999 8888 7777. Disbursement shall be credited to the account maintained at branch code HDFC0001234.

2. CO-APPLICANT AND GUARANTOR

Priya Nair Sharma has been recorded as co-applicant and shall be jointly and severally liable for repayment. The guarantee furnished by the promoter's spouse remains valid for the full tenure of the facility and may not be withdrawn without prior written consent of the Lender.

3. REPAYMENT TERMS

The facility shall be repaid in 84 equal monthly instalments commencing thirty days from the date of disbursement. Interest shall accrue at the rate specified in Schedule I and shall be calculated on a reducing balance basis. Prepayment is permitted after twelve months without penalty, subject to fifteen days written notice.

4. GENERAL

Any notice required under this Agreement shall be deemed served if delivered by registered post to the address recorded in Clause 1. This Agreement shall be governed by the laws of India and the courts at Bengaluru shall have exclusive jurisdiction. No amendment shall be effective unless made in writing and signed by both parties.
"""

FOOTER = "Synthetic sample document - all names, identifiers and addresses are fictitious."


def build_sample(out_path: str = OUT_PATH) -> str:
    doc = pymupdf.open()
    page = doc.new_page()

    page.insert_textbox(
        pymupdf.Rect(56, 50, page.rect.width - 56, 90),
        TITLE,
        fontsize=13,
        fontname="hebo",  # Helvetica-Bold
    )

    overflow = page.insert_textbox(
        pymupdf.Rect(56, 95, page.rect.width - 56, page.rect.height - 70),
        BODY,
        fontsize=10.5,
        fontname="helv",
    )
    if overflow < 0:
        raise RuntimeError(f"Body text overflowed the page by {abs(overflow):.0f} units - shorten BODY.")

    page.insert_textbox(
        pymupdf.Rect(56, page.rect.height - 62, page.rect.width - 56, page.rect.height - 40),
        FOOTER,
        fontsize=7.5,
        fontname="helv",
        color=(0.45, 0.45, 0.45),
    )

    doc.save(out_path, garbage=4, deflate=True)
    doc.close()
    return out_path


if __name__ == "__main__":
    path = build_sample()
    print(f"Wrote {path}")
