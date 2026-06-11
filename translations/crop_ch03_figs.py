"""Crop ch03 figures - round 2 with bigger search ranges."""
from PIL import Image
from pathlib import Path

PAGES = Path(r"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\work\pages")
OUT = Path(r"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\assets")

# Page is 1700 x 2200. Adjust downward.
JOBS = [
    # Figure 4: panels further down than first guess
    ("fig_03_04_search.png", 14, (100, 850, 1600, 1230)),
    # Figure 5: 2 rows extend lower
    ("fig_03_05_search.png", 15, (100, 150, 1600, 1130)),
    # Figure 6: extends below
    ("fig_03_06_search.png", 17, (100, 150, 1600, 1300)),
    # Figure 7: extends below
    ("fig_03_07_search.png", 23, (100, 150, 1600, 950)),
]

for name, page, box in JOBS:
    src = PAGES / f"page_{page:02d}.png"
    img = Image.open(src)
    out = img.crop(box)
    out.save(OUT / name)
    print(f"{name}: {out.size}")
