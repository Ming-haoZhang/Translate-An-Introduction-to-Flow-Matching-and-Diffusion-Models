"""Tighten Figure 20 to remove body text below caption."""
from PIL import Image
import numpy as np
import os

ASSETS = r"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\assets"
src = os.path.join(ASSETS, "fig_07_20.png")
img = Image.open(src)
# Crop bottom to remove the "This completes..." paragraph
# Figure caption ends around 80% of the way down
H = img.size[1]
new = img.crop((0, 0, img.size[0], int(H * 0.78)))
new.save(src)
print(f"Trimmed: {img.size} -> {new.size}")
