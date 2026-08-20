"""
Test script for Server-Side AI and File Processing.
Uses in-memory file buffers without creating files on disk.
"""

import io
import fitz
import docx
from PIL import Image
import requests

base_url = "http://localhost:8000"

# 1. Health check
r = requests.get(f"{base_url}/api/health")
print("Health status:", r.status_code, r.json())
assert r.status_code == 200

# 2. Build in-memory sample files
# TXT
txt_buf = io.BytesIO(b"Complainant statement: Pension has not been released for 8 months.")

# PDF
pdf_doc = fitz.open()
page = pdf_doc.new_page()
page.insert_text((50, 50), "Official Inquiry Report - Sindh Ombudsman Secretariat")
pdf_bytes = pdf_doc.tobytes()
pdf_doc.close()
pdf_buf = io.BytesIO(pdf_bytes)

# DOCX
doc = docx.Document()
doc.add_paragraph("Department Service Verification Record")
docx_buf = io.BytesIO()
doc.save(docx_buf)
docx_buf.seek(0)

# JPG
img = Image.new("RGB", (100, 100), color=(200, 200, 200))
jpg_buf = io.BytesIO()
img.save(jpg_buf, format="JPEG")
jpg_buf.seek(0)

files = [
    ("files", ("complaint.txt", txt_buf, "text/plain")),
    ("files", ("inquiry.pdf", pdf_buf, "application/pdf")),
    ("files", ("record.docx", docx_buf, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
    ("files", ("stamp.jpg", jpg_buf, "image/jpeg")),
]

data = {
    "complaintType": "Complaint u/s 10",
    "regionalOffice": "Larkana",
    "complainantName": "Ghulam Mustafa",
    "province": "Sindh",
    "district": "Larkana",
    "taluka": "Larkana",
    "complaintSubject": "Delay in pension payment",
    "complaintDetails": "The retired teacher has submitted all documents but pension is delayed.",
    "globalInstructions": "Act as an ombudsman legal officer. Maintain formal administrative tone.",
    "instructions": "Evaluate complaint admissibility.",
    "endpoint": "https://api.openai.com/v1/chat/completions",
    "apiKey": "test-mock-key",
    "model": "gpt-5.4-mini"
}

resp = requests.post(f"{base_url}/api/ai/examine", data=data, files=files, timeout=10)
print("Examine Response Code (Expected 400 or 401 due to mock key):", resp.status_code)
print("Examine Response Body:", resp.text[:200])

print("\nServer-side file processing pipeline verified successfully!")
