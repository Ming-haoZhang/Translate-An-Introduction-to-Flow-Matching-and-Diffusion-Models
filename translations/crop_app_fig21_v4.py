"""Crop Figure 21 v4 - cut off the figure caption text below."""
from PIL import Image
import numpy as np

img = Image.open(r"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\work\pages\page_70.png")
# Figure body only, no caption
crop = img.crop((1280, 1320, 1620, 1750))
print(f"Crop size: {crop.size}")

# trim white
arr = np.array(crop.convert("L"))
mask = arr < 240
rows = np.where(mask.any(axis=1))[0]
cols = np.where(mask.any(axis=0))[0]
if len(rows) and len(cols):
    y0, y1 = rows[0], rows[-1] + 1
    x0, x1 = cols[0], cols[-1] + 1
    pad = 15
    y0 = max(0, y0 - pad); x0 = max(0, x0 - pad)
    y1 = min(crop.size[1], y1 + pad); x1 = min(crop.size[0], x1 + pad)
    crop = crop.crop((x0, y0, x1, y1))
    print(f"Trimmed size: {crop.size}")

crop.save(r"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\assets\fig_app_21.png", optimize=True)
print("Saved fig_app_21.png v4")
