"""Crop Figure 16 from page 53 (MM-DiT architecture, in ch06)."""
import sys
sys.path.insert(0, r"C:\Users\ZhangMH\Desktop\MIT6.S184\translations")
from PIL import Image
import numpy as np

PAGE = 53
PAGE_W, PAGE_H = 1700, 2200
OUT = r"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\assets\fig_06_16.png"

# Search crop to find actual figure region
img = Image.open(fr"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\work\pages\page_{PAGE:02d}.png")
print(f"Page size: {img.size}")
# Figure 16 is at the TOP of page 53 (right after "6.3 Case Study..." header on p.53).
# Try a wide search crop
search = img.crop((0, 100, PAGE_W, int(PAGE_H * 0.5)))
search.save(fr"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\work\debug_fig16_p53_search.png")
print(f"Saved search crop: 1700x{search.size[1]}")

def trim_white(im, threshold=245, pad=18):
    a = np.array(im.convert("RGB"))
    gray = a.mean(axis=2)
    mask = gray < threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return im
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    y0 = max(0, y0 - pad); x0 = max(0, x0 - pad)
    y1 = min(a.shape[0], y1 + pad); x1 = min(a.shape[1], x1 + pad)
    return im.crop((x0, y0, x1, y1))

# From pixel scan: figure at y=247-933, header at y=129-160, caption at y=940-972
precise = img.crop((180, 240, 1530, 940))
precise.save(fr"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\work\debug_fig16_p53_precise.png")
print(f"Precise crop size: {precise.size}")

# Trim white
trimmed = trim_white(precise, threshold=245, pad=20)
trimmed.save(OUT)
print(f"Final Figure 16: {trimmed.size} -> {OUT}")
