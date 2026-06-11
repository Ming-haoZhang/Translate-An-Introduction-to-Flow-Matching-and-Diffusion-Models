"""Scan ch07 pages (p.54-64) to find Figure 17-20 exact y-coordinates by pixel scanning."""
import numpy as np
from PIL import Image
import os

PAGES_DIR = r"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\work\pages"
OUT_DIR = r"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\work"

def scan_page(p):
    """Return rows that contain non-white content and grouped bands."""
    path = os.path.join(PAGES_DIR, f"page_{p:02d}.png")
    img = np.array(Image.open(path).convert("L"))
    # Row is "content" if any pixel is below threshold
    has_content = (img < 200).any(axis=1)
    return img, has_content

def find_figure_band(p, ymin_search=200, ymax_search=2050):
    """Find horizontal bands where there's image content (consecutive content rows) in middle of page."""
    img, has_content = scan_page(p)
    H = img.shape[0]
    # Mask: in search range?
    in_range = np.zeros(H, dtype=bool)
    in_range[ymin_search:ymax_search] = has_content[ymin_search:ymax_search]
    # Find runs of True
    bands = []
    start = None
    for y in range(ymin_search, ymax_search):
        if in_range[y] and start is None:
            start = y
        elif not in_range[y] and start is not None:
            bands.append((start, y))
            start = None
    if start is not None:
        bands.append((start, ymax_search))
    return bands

# Scan pages 54-64
for p in [54, 58, 60, 65]:
    bands = find_figure_band(p, ymin_search=240, ymax_search=2050)
    print(f"\n=== p.{p} ===")
    # Merge bands that are close (< 8 px gap)
    merged = []
    for b in bands:
        if merged and b[0] - merged[-1][1] < 8:
            merged[-1] = (merged[-1][0], b[1])
        else:
            merged.append(list(b))
    # Print large bands (> 50 px = likely figure or image)
    big_bands = [b for b in merged if b[1] - b[0] > 50]
    for b in big_bands:
        h = b[1] - b[0]
        print(f"  band y={b[0]}-{b[1]} (h={h})")
