"""Re-crop ch04 Figure 8 with wider boundaries to include full caption."""
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

# Figure 8 (p.25): right column. Extend x range to capture full caption.
im = Image.open(r'C:\Users\ZhangMH\Desktop\MIT6.S184\translations\work\pages\page_25.png')
# Wider box: include "Figure 8:" prefix on left and "(right)" on right
cropped = im.crop((780, 530, 1560, 1145))
final = trim_white(cropped)
final.save(r'C:\Users\ZhangMH\Desktop\MIT6.S184\translations\assets\fig_04_08.png')
print(f"fig_04_08: {final.size} ({os.path.getsize(r'C:\Users\ZhangMH\Desktop\MIT6.S184\translations\assets\fig_04_08.png') // 1024} KB)")
