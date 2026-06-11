"""Re-crop ch05 figures with full captions."""
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

# Figure 11 (p.35): corgi images + full caption (extend y to include caption)
crop_region(
    r'C:\Users\ZhangMH\Desktop\MIT6.S184\translations\work\pages\page_35.png',
    (180, 200, 1500, 990),
    r'C:\Users\ZhangMH\Desktop\MIT6.S184\translations\assets\fig_05_11.png',
)

# Figure 12 (p.36): guidance illustration + full caption (widen x for caption text)
crop_region(
    r'C:\Users\ZhangMH\Desktop\MIT6.S184\translations\work\pages\page_36.png',
    (250, 200, 1560, 1100),
    r'C:\Users\ZhangMH\Desktop\MIT6.S184\translations\assets\fig_05_12.png',
)

# Figure 13 (p.39): MNIST grids + full caption
crop_region(
    r'C:\Users\ZhangMH\Desktop\MIT6.S184\translations\work\pages\page_39.png',
    (250, 200, 1500, 880),
    r'C:\Users\ZhangMH\Desktop\MIT6.S184\translations\assets\fig_05_13.png',
)
