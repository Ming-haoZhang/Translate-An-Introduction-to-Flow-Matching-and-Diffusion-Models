"""Re-crop Figure 17 (right column only) and Figure 20 (extend bottom for caption)."""
from PIL import Image
import numpy as np
import os

PAGES_DIR = r"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\work\pages"
ASSETS = r"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\assets"

def trim_white(img, threshold=240, pad=20):
    arr = np.array(img.convert("L"))
    H, W = arr.shape
    content = arr < threshold
    if not content.any():
        return img
    rows = np.where(content.any(axis=1))[0]
    cols = np.where(content.any(axis=0))[0]
    y0, y1 = max(0, rows[0] - pad), min(H, rows[-1] + pad)
    x0, x1 = max(0, cols[0] - pad), min(W, cols[-1] + pad)
    return img.crop((x0, y0, x1, y1))

# Figure 17: p.54 — right column only (figure on right side interleaved with text)
# Scan shows figure region is around y=1082-1444
img54 = Image.open(os.path.join(PAGES_DIR, "page_54.png"))
# Right column starts around x=875 (mid-page)
f17 = img54.crop((850, 1060, 1620, 1640))
f17 = trim_white(f17, threshold=240, pad=20)
f17.save(os.path.join(ASSETS, "fig_07_17.png"))
print(f"Figure 17: {f17.size}")

# Figure 20: p.65 — extend bottom to capture caption (y=910 -> extend to y=1820)
img65 = Image.open(os.path.join(PAGES_DIR, "page_65.png"))
f20 = img65.crop((80, 1150, 1620, 1820))
f20 = trim_white(f20, threshold=240, pad=20)
f20.save(os.path.join(ASSETS, "fig_07_20.png"))
print(f"Figure 20: {f20.size}")
