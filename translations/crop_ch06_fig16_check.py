"""Inspect precise crop without trim to see if Output is in the original."""
import sys
from PIL import Image

PAGE = 53
img = Image.open(fr"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\work\pages\page_{PAGE:02d}.png")
precise = img.crop((180, 140, 1530, 970))
precise.save(r"C:\Users\ZhangMH\Desktop\MIT6.S184\translations\work\debug_fig16_p53_notrim.png")
print(f"Saved no-trim crop: {precise.size}")
