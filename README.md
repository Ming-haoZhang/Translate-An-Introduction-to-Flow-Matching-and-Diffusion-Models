# MIT 6.S184 — 流匹配与扩散模型（中文翻译）

> **An Introduction to Flow Matching and Diffusion Models**（arXiv:2506.02070v3，Peter Holderrieth & Ezra Erives，84 页）的中文翻译版。
> 自学项目，**无课程 / 无分享截止日**，按个人节奏推进。

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)](LICENSE)
[![PDF: 84 pages](https://img.shields.io/badge/PDF-84_pages-blue.svg)](2506.02070v3.pdf)
[![arXiv:2506.02070](https://img.shields.io/badge/arXiv-2506.02070-b31b1b.svg)](https://arxiv.org/abs/2506.02070)
[![Translation: v3 (2026-06-10)](https://img.shields.io/badge/Translation-v3_(2026--06--10)-green.svg)](translations/README.md)

---

## 📑 目录

- [项目结构](#-项目结构)
- [正文（8 章主线 + 附录）](#-正文8-章主线--附录)
- [配图（22 张精裁）](#-配图22-张精裁)
- [快速开始](#-快速开始)
- [翻译规范](#-翻译规范)
- [致谢与版权](#-致谢与版权)

---

## 📂 项目结构

```
MIT6.S184/                                 # ← git 仓库根
├── README.md                              # 你正在看
├── LICENSE                                # CC BY-NC-ND 4.0
├── .gitignore
├── requirements.txt                       # pdfplumber, pypdfium2, pypdf, Pillow
├── CONTRIBUTING.md                        # 贡献指南
│
├── 2506.02070v3.pdf                       # 源论文 PDF（84 页）
│
├── scripts/                               # 工具脚本
│   ├── README.md                          # 脚本说明
│   ├── extract_pdf.py                     # 提取文本 + 渲染页面 PNG
│   └── inspect_pdf.py                     # 查看 PDF 元数据 + 大纲
│
└── translations/                          # 翻译主体（200+ KB / 8 章）
    ├── README.md                          # 翻译目录详细说明
    ├── TRANSLATION_GUIDE.md               # 翻译规范（必读）
    ├── glossary.md                        # 200+ 条统一术语表
    ├── ch01-introduction.md               # 主线章节
    ├── ch02-flow-diffusion-models.md
    ├── ch03-flow-matching.md
    ├── ch04-score-functions.md
    ├── ch05-guidance.md
    ├── ch06-large-scale-generators.md
    ├── ch07-discrete-diffusion.md
    ├── appendix.md                        # 概率复习 + 证明 + 文献指南
    └── assets/                            # 22 张精裁配图
        ├── fig_02_01.png
        ├── fig_02_02.png
        └── ...
```

> **GitHub 直接渲染**：所有 `.md` 文件中的图片路径都是相对路径（如 `assets/fig_02_01.png`），
> 在 GitHub 仓库页面上**开箱即用**。

---

## 📖 正文（8 章主线 + 附录）

| # | 章节 | PDF 页码 | 主要内容 | 关键公式 |
|---|---|---|---|---|
| 1 | [第 1 章 引言](translations/ch01-introduction.md) | p.3–6 | 流匹配、扩散、谱系定位（FM / SI / Rectified Flow） | — |
| 2 | [第 2 章 流与扩散模型](translations/ch02-flow-diffusion-models.md) | p.7–13 | ODE / SDE、流模型、Euler / Heun 数值积分 | Eq 1a–7b, Heun |
| 3 | [第 3 章 流匹配](translations/ch03-flow-matching.md) | p.14–23 | 条件 / 边缘概率路径、CFM 损失、CondOT、Algorithm 3 | Eq 10–24 |
| 4 | [第 4 章 分数函数](translations/ch04-score-functions.md) | p.25–33 | 分数函数、去噪分数匹配、Langevin、概率流 ODE | Eq 25–35 |
| 5 | [第 5 章 引导](translations/ch05-guidance.md) | p.34–39 | 分类器引导 / 无分类器引导、CFG、引导强度 | Eq 57–73 |
| 6 | [第 6 章 大规模生成器](translations/ch06-large-scale-generators.md) | p.41–53 | U-Net / DiT、VAE、Stable Diffusion 3、Movie Gen | Eq 74–95 |
| 7 | [第 7 章 离散扩散](translations/ch07-discrete-diffusion.md) | p.54–65 | CTMC、比率矩阵、掩码路径、Algorithm 7/8、MDLM | Eq 96–131 |
| — | [附录](translations/appendix.md) | p.65–84 | 51 篇参考文献 + 概率复习 + FP / CTMC 证明 + VAE 补充 | — |
| — | [术语表](translations/glossary.md) | — | 200+ 条统一术语 | — |

**总计**：~200 KB 翻译内容 + 22 张精裁配图。

---

## 🖼 配图（22 张精裁）

> 所有图都来自源 PDF，按章节 + 顺序编号。GitHub 会自动渲染下面这些相对路径图片。

| 图号 | 主题 | 位置 |
|---|---|---|
| [图 1](translations/ch02-flow-diffusion-models.md#21-流模型flow-models) | 流 $\psi_t$ 与向量场 $u_t$（微分同胚） | ch02 p.7 |
| [图 2](translations/ch02-flow-diffusion-models.md#22-随机微分方程sde) | 布朗运动轨迹 | ch02 p.10 |
| [图 3](translations/ch02-flow-diffusion-models.md) | Ornstein-Uhlenbeck 轨迹 | ch02 p.13 |
| [图 4](translations/ch03-flow-matching.md) | 噪声到数据的高斯插值 | ch03 p.14 |
| [图 5](translations/ch03-flow-matching.md) | 条件 / 边缘概率路径示意 | ch03 p.15 |
| [图 6](translations/ch03-flow-matching.md) | 玩具分布下的边缘路径 | ch03 p.18 |
| [图 7](translations/ch03-flow-matching.md) | 训练目标 vs 真实向量场 | ch03 p.22 |
| [图 8](translations/ch04-score-functions.md) | 分数函数几何含义 | ch04 p.25 |
| [图 9](translations/ch04-score-functions.md) | 分数匹配估计子 | ch04 p.30 |
| [图 10](translations/ch04-score-functions.md) | 概率流 ODE 与扩散 SDE | ch04 p.33 |
| [图 11](translations/ch05-guidance.md) | 朴素引导 vs 分类器引导 | ch05 p.35 |
| [图 12](translations/ch05-guidance.md) | CFG / CG 对比示意 | ch05 p.37 |
| [图 13](translations/ch05-guidance.md) | 引导强度 vs 样本质量 | ch05 p.39 |
| [图 14](translations/ch06-large-scale-generators.md) | U-Net 架构 | ch06 p.43 |
| [图 15](translations/ch06-large-scale-generators.md) | DiT 架构 | ch06 p.46 |
| [图 16](translations/ch06-large-scale-generators.md) | Movie Gen 系统 | ch06 p.53 |
| [图 17](translations/ch07-discrete-diffusion.md) | 离散状态空间 | ch07 p.54 |
| [图 18](translations/ch07-discrete-diffusion.md) | 掩码扩散过程 | ch07 p.58 |
| [图 19](translations/ch07-discrete-diffusion.md) | DFM 损失示意 | ch07 p.60 |
| [图 20](translations/ch07-discrete-diffusion.md) | 离散 → 连续极限 | ch07 p.65 |
| [图 21](translations/appendix.md) | 概率流守恒图示 | App.A p.70 |
| [图 22](translations/appendix.md) | VAE 潜空间示意 | App.D p.82 |

> 📌 **GitHub 锚点说明**：上面这些 `#21-流模型flow-models` 之类的链接，
> 是把中文章节标题转成 GitHub 风格的 URL anchor。
> 规则：全小写、空格变 `-`、去掉标点（`：` `（` `）` 等）。

---

## 🚀 快速开始

### 在线阅读（推荐）

直接点开仓库根目录的 [translations/README.md](translations/README.md)，
从「第 1 章 引言」开始通读。**无需任何本地工具**。

### 本地预览（VSCode / Typora / Obsidian）

```bash
git clone <this-repo>
cd MIT6.S184
code translations/ch01-introduction.md
```

> ⚠️ **公式渲染**：VSCode 推荐装 **Markdown Preview Enhanced** 或 **Markdown+Math** 扩展，
> 用 KaTeX / MathJax 渲染 `$...$` 和 `$$...$$`。

### 重新生成 PDF 文本 + 配图

```bash
pip install -r requirements.txt

# 1. 提取每页文本 + 渲染每页 PNG（200 dpi）
python scripts/extract_pdf.py
# → translations/work/text/page_NN.txt
# → translations/work/pages/page_NN.png

# 2. 裁剪配图（按章节脚本）
python translations/crop_ch02_figs.py
# → translations/assets/fig_02_0N.png
```

> 脚本里的 `PAGES_DIR` / `ASSETS_DIR` 用了**绝对路径**（Windows 风格），
> 换机器跑时需要改，或用 `pathlib.Path(__file__).parent` 改相对路径。

---

## 📝 翻译规范

- **[translations/TRANSLATION_GUIDE.md](translations/TRANSLATION_GUIDE.md)** — 翻译规范（必读）
- **[translations/glossary.md](translations/glossary.md)** — 200+ 条统一术语表
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — 贡献指南

**关键约定速览**：

- 公式严格用 LaTeX：`$...$` / `$$...$$`
- 术语首现「中文（English）」，后续只用中文；模型 / 产品名**不译**
- 中文章节标题用「第 X 章」格式
- 配图来自源 PDF，**统一白边裁剪**后存到 `translations/assets/`

> ⚠️ **关于"非法 TeX 公式"**：
> 翻译初版曾在 `\begin{aligned}` 块中嵌入 `\tag{...}`，
> 语雀（用 KaTeX 渲染）会拒绝渲染并显示"非法 TeX 公式"。
> **正确做法**是去掉 `\tag`，改用 `^{(1a)}` 文本标签。
> 详见 [CONTRIBUTING.md](CONTRIBUTING.md) 的"公式渲染约束"一节。

---

## 🛠 推荐阅读路线（自学者）

**主线数学章节**（首次通读，按顺序）：

1. ch01 → ch02 → ch03 → ch04 → ch05 → ch06 → ch07

**附录**（参考性质，对自学者特别有价值）：

- **附录 A**（概率复习）— 必备背景
- **附录 B**（Fokker-Planck 证明）— FP 方程的严格性
- **附录 C**（CTMC 证明）— ch07 的存在唯一性
- **附录 D**（VAE 补充）— LDM 架构的理论基础
- **附录 E**（文献指南）— 读原始论文的"翻译器"

**重点理解**（自学者建议放慢）：

1. **定理 3 / 9 / 12 / 17 / 22** — 连续性方程（流匹配）+ Fokker-Planck（扩散）的基础
2. **定理 23 / 25** — 条件分数 → 边缘分数的加权平均（理解去噪匹配的关键）
3. **Theorem 33 / 36 / 38** — CTMC 存在唯一性 + 边缘化 trick + 因子化条件路径下的边缘比率
4. **DFM 损失**（ch07）— cross-entropy 加权，与连续 L2 损失平行
5. **ELBO 在连续时间变紧**（App.E）— 理解为什么主线所有损失都是**等式**而非下界
6. **直觉 44**（App.D）— VAE 与潜空间生成模型的 Pareto 权衡

---

## 🙏 致谢与版权

### 源论文

> *An Introduction to Flow Matching and Diffusion Models*
> Peter Holderrieth, Ezra Erives
> arXiv:2506.02070v3, 84 pages
> [https://arxiv.org/abs/2506.02070](https://arxiv.org/abs/2506.02070)
> 协议：**CC BY-NC-ND 4.0**

本中文翻译**仅是原论文的中文化版本**，
遵守原协议的"**署名 + 非商业 + 相同方式共享**"要求。

### 本翻译项目

- **翻译**：ZhangMH（个人学习项目）
- **协议**：本仓库的中文翻译、配图裁剪脚本同样采用 **CC BY-NC-ND 4.0**（见 [LICENSE](LICENSE)）
- **版本**：v3（2026-06-10）— 全 8 章翻译完成，22 张配图精裁
- **总翻译量**：~200 KB
- **总图片**：22 张

### 使用限制

✅ 可以：
- 自由复制、分享本翻译
- 用于个人学习

❌ 不得：
- 用于任何商业目的
- 二次创作 / 修改后重新发布
- 移除原作者署名

如需用于商业或二次发布，请联系作者获取单独许可。

---

## 📬 反馈

- **公式 / 翻译错误**：开 Issue，标注章节号 + 行号
- **术语建议**：参考 [glossary.md](translations/glossary.md) 后开 PR
- **大改 / 整章重译**：先开 Issue 讨论
