"""Crop Figure 22 from p.82 (gFID-rFID tradeoff + distortion-rate curve)."""
from PIL import Image

img = Image.open(r"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\work\pages\page_82.png")
print(f"Page size: {img.size}")

# Figure 22 occupies the top portion of p.82, roughly y=180 to y=620
# left margin x=240, right margin x=240
crop = img.crop((240, 180, 1700 - 240, 700))
print(f"Crop size: {crop.size}")

# trim white
import numpy as np
arr = np.array(crop.convert("L"))
mask = arr < 240
rows = np.where(mask.any(axis=1))[0]
cols = np.where(mask.any(axis=0))[0]
if len(rows) and len(cols):
    y0, y1 = rows[0], rows[-1] + 1
    x0, x1 = cols[0], cols[-1] + 1
    pad = 20
    y0 = max(0, y0 - pad); x0 = max(0, x0 - pad)
    y1 = min(crop.size[1], y1 + pad); x1 = min(crop.size[0], x1 + pad)
    crop = crop.crop((x0, y0, x1, y1))
    print(f"Trimmed size: {crop.size}")

crop.save(r"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\assets\fig_app_22.png", optimize=True)
print("Saved fig_app_22.png")
