# scripts/

PDF 处理与配图裁剪工具脚本。

## 顶层脚本（项目根目录脚本）

| 脚本 | 用途 | 输入 | 输出 |
|---|---|---|---|
| `extract_pdf.py` | 从源 PDF 提取每页文本 + 渲染每页 PNG | `../2506.02070v3.pdf` | `../translations/work/text/page_NN.txt` + `../translations/work/pages/page_NN.png` |
| `inspect_pdf.py` | 查看 PDF 元数据 + 大纲 | `../2506.02070v3.pdf` | stdout（页数 / 元数据 / 章节书签） |

## 各章配图裁剪脚本

> 这些脚本在 [translations/](../translations/) 目录下，**与翻译章节同放**，
> 便于维护时一眼看到"哪个脚本裁哪一章的图"。

主要脚本（按章节）：

- `translations/crop_ch02_figs.py` — ch02 三张图（流、布朗轨迹、Ornstein-Uhlenbeck）
- `translations/crop_ch03_figs.py` — ch03 四张图
- `translations/crop_ch03_figs_precise.py` — ch03 配图精修版
- `translations/crop_ch04_figs.py` — ch04 三张图
- `translations/crop_ch05_figs.py` / `crop_ch05_figs_v2.py` — ch05 三张图
- `translations/crop_ch06_figs.py` — ch06 配图
- `translations/crop_ch06_fig16.py` — ch06 图 16 单独裁剪
- `translations/crop_ch06_fig16_check.py` — 图 16 校验
- `translations/crop_ch07_figs.py` / `crop_ch07_figs_v2.py` / `crop_ch07_fig20_tight.py` — ch07 配图
- `translations/crop_app_fig21.py` 及其 v2/v3/v4 — 附录图 21 多版本迭代
- `translations/crop_app_fig22.py` / `crop_app_fig22_v2.py` — 附录图 22
- `translations/crop_fig08_fix.py` — ch04 图 8 修复
- `translations/find_header.py` — 工具：定位 PDF 页眉
- `translations/scan_ch07_figs.py` — 工具：扫描 ch07 配图位置

> 脚本里的 `PAGES_DIR` / `ASSETS_DIR` 是绝对路径（`C:\Users\ZhangMH\Desktop\MIT6.S184\...`），
> 在其他机器上跑时需要改路径，或用 `pathlib.Path(__file__).parent` 改成相对。

## 安装

```bash
pip install -r ../requirements.txt
```

## 典型流程

```bash
# 1. 从 PDF 提取文本 + 渲染页面
python scripts/extract_pdf.py

# 2. 裁剪第二章配图
python translations/crop_ch02_figs.py
# → 输出到 translations/assets/fig_02_0N.png
```
