"""Find actual y position of page header on p.53."""
from PIL import Image
import numpy as np

img = Image.open(r"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\work\pages\page_53.png")
a = np.array(img.convert("L"))  # grayscale
H, W = a.shape
print(f"Page: {W}x{H}")

# Find horizontal text bands: rows that have dark pixels (text)
# A row is "text" if it has many dark pixels in the central region
text_rows = []
for y in range(H):
    row = a[y, 200:1500]  # central region
    dark = (row < 200).sum()
    if dark > 20:  # threshold for "has text"
        text_rows.append((y, dark))

# Find bands (consecutive text rows)
bands = []
if text_rows:
    cur_start = text_rows[0][0]
    cur_end = cur_start
    for y, dark in text_rows[1:]:
        if y - cur_end <= 5:
            cur_end = y
        else:
            bands.append((cur_start, cur_end))
            cur_start = y
            cur_end = y
    bands.append((cur_start, cur_end))

print(f"Found {len(bands)} text bands in central region:")
for i, (s, e) in enumerate(bands[:20]):
    print(f"  Band {i}: y={s}-{e} (h={e-s+1})")
