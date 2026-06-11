# MIT 6.S184 流匹配与扩散模型 — 中文翻译

> 完整中文化 *An Introduction to Flow Matching and Diffusion Models*（arXiv:2506.02070v3，Peter Holderrieth & Ezra Erives，CC BY-NC-ND 4.0，84 页）。
> 自学项目，无课程 / 分享截止日，按个人节奏推进。

## 📚 目录

| # | 文件 | PDF 页码 | 大小 | 主要内容 |
|---|---|---|---|---|
| 1 | [ch01-introduction.md](ch01-introduction.md) | p.3–6 | 11.2 KB | 引言：流匹配、扩散、Flow Matching / SI / Rectified Flow 谱系 |
| 2 | [ch02-flow-diffusion-models.md](ch02-flow-diffusion-models.md) | p.7–13 | 19.8 KB | ODE/SDE、流模型、扩散模型、Euler / Heun 数值积分 |
| 3 | [ch03-flow-matching.md](ch03-flow-matching.md) | p.14–23 | 31.8 KB | 条件 / 边缘概率路径、CFM 损失、CondOT、Algorithm 3 |
| 4 | [ch04-score-functions.md](ch04-score-functions.md) | p.25–33 | 22.2 KB | 分数函数、去噪分数匹配、Langevin、概率流 ODE |
| 5 | [ch05-guidance.md](ch05-guidance.md) | p.34–39 | 15.7 KB | 分类器引导 / 无分类器引导、CFG、引导强度 |
| 6 | [ch06-large-scale-generators.md](ch06-large-scale-generators.md) | p.41–53 | 38.4 KB | 神经网络架构（U-Net / DiT）、VAE、Stable Diffusion 3、Movie Gen |
| 7 | [ch07-discrete-diffusion.md](ch07-discrete-diffusion.md) | p.54–65 | 23.4 KB | CTMC、比率矩阵、掩码路径、Algorithm 7/8、MDLM |
| 8 | [appendix.md](appendix.md) | p.65–84 | 34.8 KB | 51 篇参考文献 + 概率复习 + FP / CTMC 证明 + VAE 补充 + 文献指南 |
| — | [glossary.md](glossary.md) | — | — | 统一术语表（200+ 条目） |
| — | [TRANSLATION_GUIDE.md](TRANSLATION_GUIDE.md) | — | — | 翻译规范 |

**总计**：~200 KB 翻译内容 + 22 张精裁配图

## 🖼 配图清单

| 图号 | 文件 | 位置 | 大小 |
|---|---|---|---|
| 图 1 | `assets/fig_02_01.png` | ch02 p.7 | 1410×315 |
| 图 2 | `assets/fig_02_02.png` | ch02 p.10 | 490×580 |
| 图 3 | `assets/fig_02_03.png` | ch02 p.13 | 1435×400 |
| 图 4 | `assets/fig_03_04.png` | ch03 p.14 | 1428×380 |
| 图 5 | `assets/fig_03_05.png` | ch03 p.15 | 1416×980 |
| 图 6 | `assets/fig_03_06.png` | ch03 p.18 | 1415×1150 |
| 图 7 | `assets/fig_03_07.png` | ch03 p.22 | 1423×800 |
| 图 8 | `assets/fig_04_08.png` | ch04 p.25 | 777×576 |
| 图 9 | `assets/fig_04_09.png` | ch04 p.30 | 1417×1081 |
| 图 10 | `assets/fig_04_10.png` | ch04 p.33 | 1416×965 |
| 图 11 | `assets/fig_05_11.png` | ch05 p.35 | 1320×780 |
| 图 12 | `assets/fig_05_12.png` | ch05 p.37 | 1306×890 |
| 图 13 | `assets/fig_05_13.png` | ch05 p.39 | 1250×630 |
| 图 14 | `assets/fig_06_14.png` | ch06 p.43 | 1300×530 |
| 图 15 | `assets/fig_06_15.png` | ch06 p.46 | 1440×630 |
| 图 16 | `assets/fig_06_16.png` | ch06 p.53 | 1137×700 |
| 图 17 | `assets/fig_07_17.png` | ch07 p.54 | 709×523 |
| 图 18 | `assets/fig_07_18.png` | ch07 p.58 | 1421×594 |
| 图 19 | `assets/fig_07_19.png` | ch07 p.60 | 1418×753 |
| 图 20 | `assets/fig_07_20.png` | ch07 p.65 | 1418×491 |
| 图 21 | `assets/fig_app_21.png` | App.A p.70 | 271×334 |
| 图 22 | `assets/fig_app_22.png` | App.D p.82 | 1357×472 |

## 🎯 自学建议（与本翻译配套）

主线章节是**数学主线**——首次通读建议按顺序 ch01 → ch07，每章配图配合定理 / 公式一起读。**附录**是参考性质，对**自学者**特别有价值：
- **附录 A**（概率复习）：必备背景
- **附录 B**（Fokker-Planck 证明）：理解 Fokker-Planck 方程的严格性
- **附录 C**（CTMC 证明）：理解 ch07 的存在唯一性
- **附录 D**（VAE 补充）：理解 LDM 架构的理论基础
- **附录 E**（文献指南）：读原始论文的"翻译器"

**关键数学点（建议重点理解）**：

1. **定理 9 / 12 / 17 / 22**：连续性方程（流匹配）+ Fokker-Planck（扩散）的基础
2. **定理 23 / 25**：条件分数 → 边缘分数的加权平均（理解去噪匹配的关键）
3. **Theorem 33 / 36 / 38**：CTMC 的存在唯一性 + 边缘化 trick + 因子化条件路径下的边缘比率
4. **DFM 损失**（ch07）：cross-entropy 加权，与连续 L2 损失平行
5. **ELBO 在连续时间变紧**（App.E）：理解为什么主线所有损失都是**等式**而非下界
6. **直觉 44**（App.D）：VAE 与潜空间生成模型的 Pareto 权衡

## 🛠 翻译技术细节

- 公式严格用 LaTeX（行内 `$...$` / 独立 `$$...$$`）
- 配图来自源 PDF（200 dpi 渲染 + 像素扫描 + 白边自动裁剪）
- 术语首现中文（English），后续只用中文；模型 / 产品名不译
- 中文章节标题用「第 X 章」格式，附录用「附录」

## 📜 版权声明

源论文 *An Introduction to Flow Matching and Diffusion Models* by Peter Holderrieth & Ezra Erives 遵循 **CC BY-NC-ND 4.0** 协议。本中文翻译**仅供个人学习使用**，未做任何商业用途。如需二次发布，请遵守原协议（署名、非商业、相同方式共享）。

---

**版本**：v3（2026-06-10）
**总翻译量**：~200 KB
**总图片**：22 张
