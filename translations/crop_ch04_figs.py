"""Crop ch04 figures (8, 9, 10) - corrected boundaries."""
from PIL import Image
import os

def trim_white(img, threshold=245, pad=18):
    import numpy as np
    arr = np.array(img.convert('RGB'))
    mask = (arr < threshold).any(axis=2)
    if not mask.any():
        return img
    rows = mask.any(axis=1)
    cols = mask.any(axis=0)
    y0, y1 = rows.argmax(), len(rows) - rows[::-1].argmax()
    x0, x1 = cols.argmax(), len(cols) - cols[::-1].argmax()
    y0 = max(0, y0 - pad); x0 = max(0, x0 - pad)
    y1 = min(img.height, y1 + pad); x1 = min(img.width, x1 + pad)
    return img.crop((x0, y0, x1, y1))

def crop_region(src, box, out):
    im = Image.open(src)
    cropped = im.crop(box)
    final = trim_white(cropped)
    final.save(out)
    print(f"{out}: {final.size} ({os.path.getsize(out) // 1024} KB)")

# Figure 8 (p.25): right column panels + caption. The figure is at right side
# of the page; left panel starts around x=820, right panel ends around x=1500.
# Skip the text above. y starts around 530.
crop_region(
    r'C:\Users\ZhangMH\Desktop\MIT6.S184\translations\work\pages\page_25.png',
    (820, 530, 1500, 1100),
    r'C:\Users\ZhangMH\Desktop\MIT6.S184\translations\assets\fig_04_08.png',
)

# Figure 9 (p.27): top half of the page, including both rows of panels and caption
crop_region(
    r'C:\Users\ZhangMH\Desktop\MIT6.S184\translations\work\pages\page_27.png',
    (100, 200, 1600, 1330),
    r'C:\Users\ZhangMH\Desktop\MIT6.S184\translations\assets\fig_04_09.png',
)

# Figure 10 (p.30): top half of the page
crop_region(
    r'C:\Users\ZhangMH\Desktop\MIT6.S184\translations\work\pages\page_30.png',
    (100, 200, 1600, 1180),
    r'C:\Users\ZhangMH\Desktop\MIT6.S184\translations\assets\fig_04_10.png',
)
