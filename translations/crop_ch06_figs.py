"""Crop ch06 figures: Figure 14 (p.43) and Figure 15 (p.46)."""
from PIL import Image
import os

PAGES_DIR = r"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\work\pages"
OUT_DIR = r"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\assets"
os.makedirs(OUT_DIR, exist_ok=True)


def trim_white(img, threshold=245, pad=18):
    """Trim white margins. threshold: minimum brightness to consider 'white'."""
    gray = img.convert("L")
    bw = gray.point(lambda p: 0 if p < threshold else 255)
    bbox = bw.getbbox()
    if bbox is None:
        return img
    l, t, r, b = bbox
    l = max(0, l - pad)
    t = max(0, t - pad)
    r = min(img.width, r + pad)
    b = min(img.height, b + pad)
    return img.crop((l, t, r, b))


def search_crop(page_path, y_start_norm, y_end_norm, x_start_norm=0.05, x_end_norm=0.95, label=""):
    """Crop a region by normalized coordinates and save as search image."""
    img = Image.open(page_path)
    w, h = img.size
    box = (int(x_start_norm * w), int(y_start_norm * h),
           int(x_end_norm * w), int(y_end_norm * h))
    print(f"[{label}] page size: {w}x{h}, box: {box}")
    cropped = img.crop(box)
    out_search = os.path.join(OUT_DIR, f"_search_{label}.png")
    cropped.save(out_search)
    print(f"  saved search: {out_search} ({cropped.size})")
    return cropped


def precise_crop(page_path, x1, y1, x2, y2, out_name, threshold=245, pad=18):
    """Crop precise pixel coordinates, then trim white."""
    img = Image.open(page_path)
    print(f"[{out_name}] page size: {img.size}, box: ({x1},{y1},{x2},{y2})")
    cropped = img.crop((x1, y1, x2, y2))
    trimmed = trim_white(cropped, threshold=threshold, pad=pad)
    out_path = os.path.join(OUT_DIR, out_name)
    trimmed.save(out_path)
    print(f"  saved: {out_path} ({trimmed.size})")


if __name__ == "__main__":
    page_43 = os.path.join(PAGES_DIR, "page_43.png")
    page_46 = os.path.join(PAGES_DIR, "page_46.png")

    # First: search crops to confirm figure bounds
    # search_crop(page_43, 0.05, 0.42, label="fig14_p43")
    # search_crop(page_46, 0.05, 0.42, label="fig15_p46")

    # Precise crop: based on search crops, figure bodies are roughly:
    #   Figure 14 (p.43): y=240-770 (caption starts ~y=780)
    #   Figure 15 (p.46): y=240-870 (full Downsample label visible)
    precise_crop(page_43, 180, 240, 1480, 770, "fig_06_14.png")
    precise_crop(page_46, 120, 240, 1560, 870, "fig_06_15.png")
