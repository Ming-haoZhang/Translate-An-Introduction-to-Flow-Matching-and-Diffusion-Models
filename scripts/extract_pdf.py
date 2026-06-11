"""Extract per-page text + render per-page PNG from the PDF.
Output:
  work/text/page_NN.txt
  work/pages/page_NN.png  (200 dpi)
"""
from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium

PDF = Path(r"C:\Users\ZhangMH\Desktop\MIT6.S184\2506.02070v3.pdf")
TXT_DIR = Path(r"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\work\text")
PNG_DIR = Path(r"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\work\pages")
TXT_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR.mkdir(parents=True, exist_ok=True)

# ---- 1. Extract text per page (layout-preserving) ----
print("=== Extracting text ===")
with pdfplumber.open(PDF) as pdf:
    for i, page in enumerate(pdf.pages, start=1):
        txt = page.extract_text(layout=True) or ""
        (TXT_DIR / f"page_{i:02d}.txt").write_text(txt, encoding="utf-8")
print(f"Wrote {len(list(TXT_DIR.glob('*.txt')))} text files")

# ---- 2. Render every page to PNG at 200 dpi ----
print("\n=== Rendering pages to PNG ===")
pdf_doc = pdfium.PdfDocument(str(PDF))
n = len(pdf_doc)
for i in range(n):
    page = pdf_doc[i]
    bitmap = page.render(scale=200 / 72)  # 200 dpi
    img = bitmap.to_pil()
    out = PNG_DIR / f"page_{i + 1:02d}.png"
    img.save(out, "PNG", optimize=True)
    if (i + 1) % 10 == 0 or i == 0:
        print(f"  rendered {i + 1}/{n} -> {out.name}  ({img.size[0]}x{img.size[1]})")
pdf_doc.close()
print(f"\nDone. {n} pages rendered to {PNG_DIR}")
