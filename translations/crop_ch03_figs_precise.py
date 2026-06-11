"""精裁 ch03 的 4 张图：基于 search crop 自动检测非白像素边界，加 padding。"""
from PIL import Image
import numpy as np
import os

ASSETS = r"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\assets"


def trim_white(img: Image.Image, threshold: int = 245, pad: int = 12) -> Image.Image:
    """检测非白像素 (RGB 中任一 < threshold) 的精确边界, 加 pad 像素。"""
    arr = np.array(img.convert("RGB"))
    # 找"不是白纸"的像素: 灰度化 + 阈值
    gray = arr.min(axis=2)  # 任何通道低于阈值都算
    mask = gray < threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        raise ValueError("No non-white content found")
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(arr.shape[0], y1 + pad)
    x1 = min(arr.shape[1], x1 + pad)
    return img.crop((x0, y0, x1, y1))


# 精裁 4 张图
tasks = [
    # (search 文件, 输出文件)
    ("fig_03_04_search.png", "fig_03_04.png"),
    ("fig_03_05_search.png", "fig_03_05.png"),
    ("fig_03_06_search.png", "fig_03_06.png"),
    ("fig_03_07_search.png", "fig_03_07.png"),
]

for src, dst in tasks:
    src_path = os.path.join(ASSETS, src)
    dst_path = os.path.join(ASSETS, dst)
    img = Image.open(src_path)
    print(f"{src}: input size = {img.size}")
    trimmed = trim_white(img, threshold=245, pad=18)
    trimmed.save(dst_path, optimize=True)
    print(f"  -> {dst}: {trimmed.size}  ({os.path.getsize(dst_path) // 1024} KB)")
