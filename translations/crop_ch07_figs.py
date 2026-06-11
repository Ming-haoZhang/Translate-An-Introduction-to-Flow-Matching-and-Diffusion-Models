"""Crop ch07 Figure 17, 18, 19, 20 with precise y-coordinates from pixel scan."""
import numpy as np
from PIL import Image
import os

PAGES_DIR = r"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\work\pages"
ASSETS = r"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\assets"

def trim_white(img, threshold=240, pad=20):
    """Trim white margins from a PIL image."""
    arr = np.array(img.convert("L"))
    H, W = arr.shape
    # Find non-white content
    content = arr < threshold
    if not content.any():
        return img
    rows = np.where(content.any(axis=1))[0]
    cols = np.where(content.any(axis=0))[0]
    y0, y1 = max(0, rows[0] - pad), min(H, rows[-1] + pad)
    x0, x1 = max(0, cols[0] - pad), min(W, cols[-1] + pad)
    return img.crop((x0, y0, x1, y1))

# Crops: (page, output_name, (x0, y0, x1, y1))
crops = [
    # Figure 17: p.54, CTMC trajectory with state space S = S1,S2,S3 (sequence length d=1)
    # Right-side figure interleaved with text; full page-width crop covers it
    (54, "fig_07_17.png", (80, 1050, 1620, 1620)),
    # Figure 18: p.58, factorized CTMC model, General vs Factorized rate matrix
    # Big band y=284-708, caption y=754+
    (58, "fig_07_18.png", (80, 240, 1620, 880)),
    # Figure 19: p.60, discrete probability path d=2
    # Two stacked rows y=843-1363, caption y=1464+
    (60, "fig_07_19.png", (80, 810, 1620, 1610)),
    # Figure 20: p.65, MDLM trajectory (sequence table)
    # y=943-1546 with caption
    (65, "fig_07_20.png", (80, 910, 1620, 1620)),
]

for page, name, box in crops:
    path = os.path.join(PAGES_DIR, f"page_{page:02d}.png")
    img = Image.open(path)
    print(f"p.{page}: full {img.size}, cropping {box}")
    cropped = img.crop(box)
    # Trim white
    trimmed = trim_white(cropped, threshold=240, pad=20)
    out = os.path.join(ASSETS, name)
    trimmed.save(out)
    print(f"  -> {out} ({trimmed.size})")
