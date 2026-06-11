"""Inspect the PDF: page count, metadata, and table of contents (bookmarks)."""
import json
import sys
from pathlib import Path

import pdfplumber

PDF_PATH = r"C:\Users\ZhangMH\Desktop\MIT6.S184\2506.02070v3.pdf"

with pdfplumber.open(PDF_PATH) as pdf:
    print(f"Pages: {len(pdf.pages)}")
    print(f"Metadata: {pdf.metadata}")

# Get bookmarks/outline using pypdf
from pypdf import PdfReader

reader = PdfReader(PDF_PATH)
print(f"\n=== Outline / Bookmarks ===")
def walk_outline(items, depth=0):
    out = []
    if not items:
        return out
    try:
        for item in items:
            if isinstance(item, list):
                out.extend(walk_outline(item, depth + 1))
            else:
                try:
                    page = reader.get_destination_page_number(item)
                except Exception:
                    page = "?"
                out.append({
                    "depth": depth,
                    "title": str(item.title),
                    "page": page,
                })
    except Exception as e:
        print(f"  outline walk error: {e}")
    return out

outline = walk_outline(reader.outline)
for entry in outline:
    indent = "  " * entry["depth"]
    print(f"{indent}p.{entry['page']:>4}  {entry['title']}")
print(f"\nTotal outline entries: {len(outline)}")
