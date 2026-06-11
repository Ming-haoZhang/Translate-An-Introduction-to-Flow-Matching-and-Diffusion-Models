"""Crop figures from rendered PDF page PNGs for the translations.

PNG source: translations/work/pages/page_NN.png (1700x2200 at 200 dpi)
Output:     translations/assets/fig_chXX_NN.png
"""

import os
from PIL import Image

PAGES_DIR = r"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\work\pages"
ASSETS_DIR = r"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\assets"

os.makedirs(ASSETS_DIR, exist_ok=True)


def crop(page_num: int, out_name: str, box):
    """box = (left, top, right, bottom) in original pixel coords."""
    src = os.path.join(PAGES_DIR, f"page_{page_num:02d}.png")
    img = Image.open(src)
    print(f"page {page_num}: {img.size} -> crop {box}")
    cropped = img.crop(box)
    out = os.path.join(ASSETS_DIR, out_name)
    cropped.save(out, "PNG", optimize=True)
    print(f"  saved {out}  ({cropped.size})")


# Figure 1 (page 8): flow + vector field grid, top of page
crop(8, "fig_02_01.png", (155, 165, 1565, 480))

# Figure 2 (page 10): Brownian motion trajectories, right column
crop(10, "fig_02_02.png", (960, 720, 1450, 1300))

# Figure 3 (page 11): Ornstein-Uhlenbeck trajectories for various sigma, top of page
crop(11, "fig_02_03.png", (135, 200, 1570, 600))

print("done.")
