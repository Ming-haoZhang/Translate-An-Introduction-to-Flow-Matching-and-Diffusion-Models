# 翻译规范（Translation Conventions）

本文档是 MIT 6.S184 / arXiv:2506.02070v3 — *An Introduction to Flow Matching and Diffusion Models* 中文翻译的**统一规范**，所有翻译者（无论是我还是 mavis-team 派出的 worker）都必须遵守。

---

## 1. 通用规则

| 规则 | 说明 |
|---|---|
| **行文风格** | 学术中文，语气客观。不滥用"我们"作为代词；保留"我们"指代作者团队。避免出现"我"作为主语。 |
| **段落结构** | 完全对应原文：一个英文段落 → 一个中文段落，不合并、不拆分。 |
| **粗体/斜体** | 保留原文的 `\textbf{}` / `\textit{}` 视觉强调，翻译后用 `**...**` / `*...*`。 |
| **链接** | 原文的蓝色引用链接（章节引用、参考文献）保留，必要时改成中文版内链。 |
| **不译** | 模型名（Stable Diffusion 3、Nano Banana、ChatGPT、AlphaFold3、FLUX、VEO-3、SORA）、论文标题、作者名、引用编号 `[43]`、英文产品名/项目名。 |

---

## 2. 数学公式

**所有数学公式严格用 LaTeX。**

- 行内公式：`$...$`
- 独立公式：`$$...$$`
- 多行对齐公式：用 `\begin{aligned} ... \end{aligned}`，并用 `\\` 换行，`&` 对齐。
- 不要把公式翻译成中文变量名 — **保留原文符号**。例如：
  - `$X_t \sim p_{\text{data}}$` 不要改成「$X_t$ 服从数据分布」。
  - 但行文描述可以加中文：「$X_t$ 服从数据分布 $p_{\text{data}}$」。

复杂公式模板：

```markdown
$$
\begin{aligned}
X_0 &\sim p_{\text{init}} \\
dX_t &= u_t^{\theta}(X_t)\,dt + \sigma_t\,dW_t
\end{aligned}
$$
```

---

## 3. 章节与目录

按 PDF outline 切分章节（注意 outline 的页码是 PDF 实际页码 = `page_XX.png` 的编号）：

| 文件 | 内容 | PDF 起始页 |
|---|---|---|
| `ch01-introduction.md` | 1 Introduction | 3 |
| `ch02-flow-diffusion-models.md` | 2 Flow and Diffusion Models | 6 |
| `ch03-flow-matching.md` | 3 Flow Matching | 13 |
| `ch04-score-matching.md` | 4 Score Functions and Score Matching | 24 |
| `ch05-guidance.md` | 5 Guidance | 33 |
| `ch06-large-scale-generators.md` | 6 Building Large-Scale Image/Video Generators | 40 |
| `ch07-discrete-diffusion.md` | 7 Discrete Diffusion Models | 53 |
| `appendix.md` | 8 References + Appendices A-E | 65-83 |

每章 md 文件开头用以下格式：

```markdown
# 第 X 章 [中文标题]

> 原文：[An Introduction to Flow Matching and Diffusion Models](https://arxiv.org/abs/2506.02070)
> 作者：Peter Holderrieth, Ezra Erives
> 章节页码：PDF p.X-Y

## X.1 [小节中文标题]
...
```

---

## 4. 图片处理

PDF 中所有**矢量图**（流程图、坐标图、示意图）用 `pdftoppm` 把整页截下来后**手动裁剪**到图区域，存为 `assets/fig_XX_YY.png`：

- `XX` = 章节号（两位）
- `YY` = 图在该章节中的序号（两位，从 01 起）

在 md 中引用：

```markdown
![图 X.Y 说明](assets/fig_XX_YY.png)
*图 X.Y: 中文图注。*
```

如某页**只有图、没有可译文字**，依然要把图嵌入（用整页 PNG + 中文图注）。

---

## 5. 特殊块的翻译

原文的 "Key Idea" / "Summary" / "Remark" / "Definition" / "Algorithm" 等带框的块，翻译时用引用 + 粗体表示：

```markdown
> **关键思想 1（Objects as Vectors / 对象即向量）**
> 
> 我们把待生成的对象识别为向量 $z \in \mathbb{R}^d$。
```

> 注：原文中带 `(...)` 的副标题翻译为「/」分隔，例如 `(Objects as Vectors)` → 「对象即向量」。如果副标题意译不顺，保留英文原文。

---

## 6. 术语表（部分）

完整术语表见 `glossary.md`。下面是**首次出现时**统一翻译方式（括号内给英文原词）：

| 英文 | 中文 | 备注 |
|---|---|---|
| flow model | 流模型 | |
| diffusion model | 扩散模型 | |
| denoising diffusion model | 去噪扩散模型 | |
| flow matching | 流匹配 | |
| score function | 分数函数 | 即 $\nabla \log p_t$ |
| score matching | 分数匹配 | |
| ordinary differential equation (ODE) | 常微分方程 | |
| stochastic differential equation (SDE) | 随机微分方程 | |
| probability path | 概率路径 | |
| conditional probability path | 条件概率路径 | |
| marginal probability path | 边缘概率路径 | |
| vector field | 向量场 | |
| conditional vector field | 条件向量场 | |
| marginal vector field | 边缘向量场 | |
| data distribution | 数据分布 | $p_{\text{data}}$ |
| prior distribution | 先验分布 | |
| noise schedule | 噪声调度 | |
| classifier-free guidance | 无分类器引导 | |
| classifier guidance | 分类器引导 | |
| latent space | 潜空间 | |
| variational autoencoder (VAE) | 变分自编码器 | |
| continuous-time Markov chain (CTMC) | 连续时间马尔可夫链 | |
| discrete diffusion | 离散扩散 | |
| neural network | 神经网络 | |
| U-Net | U-Net | 不译 |
| Transformer | Transformer | 不译 |
| diffusion coefficient | 扩散系数 | |
| Euler-Maruyama method | 欧拉-丸山方法 | |
| Fokker-Planck equation | 福克-普朗克方程 | |
| probability density | 概率密度 | |
| likelihood | 似然 | |
| frames | 帧 | 视频帧 |
| modality / modalities | 模态 | |
| conditioning | 条件化 / 条件作用 | |
| prompt | 提示 | 在 diffusion 中常指文本条件 |

> **首现原则**：术语第一次出现时写「中文（English）」，后续只用中文。但对模型/产品名/方法名（如 flow matching、score matching）第一次出现给中文 + 英文，**后续英文不再重复**。
