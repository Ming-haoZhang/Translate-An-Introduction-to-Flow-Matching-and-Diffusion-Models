# 第 1 章 引言（Introduction）

> 原文：[*An Introduction to Flow Matching and Diffusion Models*](https://arxiv.org/abs/2506.02070) by Peter Holderrieth & Ezra Erives
> 章节页码：PDF p.3–6

---

> *Creating noise from data is easy; creating data from noise is generative modeling.*
> （从数据制造噪声是容易的；从噪声制造数据才是生成式建模。）
>
> —— Song et al. [43]

## 1.1 概述（Overview）

近年来，我们共同见证了人工智能（AI, Artificial Intelligence）领域的一场巨大变革。图像生成器如 *Nano Banana*、*Stable Diffusion 3* 能够跨多种风格生成照片级真实感和艺术化的图像；视频模型如 Meta 的 *VEO-3* 能够生成高度逼真的电影片段；大语言模型如 *ChatGPT* 能够针对文本提示生成看似人类水平的回答。驱动这场变革的核心，是 AI 系统所获得的一种新能力——**生成（generate）**对象的能力。早期 AI 系统主要用于**预测（prediction）**，而新一代 AI 系统则更具创造性：它们能根据用户指定的输入"想象"或创造新对象。这样的**生成式 AI（generative AI）**系统正是这场 AI 变革的核心。

本课程的目标是教授两种应用最广泛的生成式 AI 算法：**去噪扩散模型（denoising diffusion models）** [45] 与**流匹配（flow matching）** [25, 27, 1, 26]。这些模型是当前最优秀的图像、音频、视频生成模型（如 *Nano Banana*、*FLUX*、*VEO-3*）的骨干，并已成为科学应用（如蛋白质结构——*AlphaFold3* 就是一个扩散模型）中的最新技术。毫无疑问，深入理解这些模型是一项极具实用价值的技能。

所有这些生成模型都通过**迭代地将噪声（noise）转化为数据（data）**来生成对象。这种"从噪声到数据"的演化过程是通过**常微分方程或随机微分方程（ODEs/SDEs, ordinary or stochastic differential equations）**的数值模拟来实现的。流匹配与去噪扩散模型是一族允许我们在深度神经网络支持下大规模**构造、训练并模拟**这类 ODE/SDE 的方法。这些模型虽然实现起来相对简单，但 SDE 本身的数学性质使它们较难理解。本课程将提供关于微分方程所需数学工具箱的自包含式介绍，帮助你系统地理解这些模型。随后我们会一步步讲解现代最先进（state-of-the-art）图像与视频生成器的完整技术栈。除了广泛的实用性外，我们相信流与扩散模型背后的理论本身也很优雅。所以，最重要的，我们希望这门课能给你带来很多乐趣。

> **备注 1（Additional Resources / 补充资源）**
> 
> 尽管本讲义是自包含的，我们仍建议大家使用以下两类补充资源：
> 
> 1. **Lecture recordings（讲堂录像）**：以授课形式带你走遍每一节内容。
> 2. **Labs（实验）**：引导你从零实现一个自己的扩散模型。强烈建议你"亲自动手"写代码。
> 
> 你可以在课程网站上找到这些资源：<https://diffusion.csail.mit.edu/>。

## 1.2 课程结构（Course Structure）

下面给出本文档的简要导览。

- **第 1 节，生成式建模即采样（Generative Modeling as Sampling）**：我们把"生成"一张图像、一段视频、一个蛋白质等任务形式化。我们会把"如何生成一张狗的图像？"这类问题翻译成更精确的"从一个概率分布中采样"的问题。

- **第 2 节，流与扩散模型（Flow and Diffusion Models）**：我们解释"生成"的机制。顾名思义，这一机制的核心是**常微分方程与随机微分方程的数值模拟**。我们介绍微分方程的基础，并展示如何利用它们构建生成模型。

- **第 3 节，流匹配（Flow Matching）**：接下来我们推导并解释 **flow matching**——一种简单且可扩展的算法，它正是 Stable Diffusion、Nano Banana、SORA 等所有前述大规模生成模型的核心。

- **第 4 节，分数匹配（Score Matching）**：我们研究**分数函数（score functions）**及其通过 **score matching** 学习的途径。它不仅作为扩散模型的训练算法，还打开了 SDE 采样和引导（guidance）的大门。

- **第 5 节，引导（Guidance）**：我们学习如何让采样过程以一个提示（prompt）（例如"一张猫的图像"）为条件，以及如何通过 classifier-free guidance 强化对提示的遵循。

- **第 6 节，潜空间与神经网络架构（Latent Spaces, Neural Network architectures）**：我们讨论如何构建像 *Nano Banana* 这样的大规模图像/视频生成器。内容包括常见的神经网络架构，以及如何在潜空间中构建模型。我们还会概览当前的 state-of-the-art 模型。

- **第 7 节（可选），离散扩散模型（Discrete Diffusion Models）**：我们学习如何把扩散模型的原理从欧几里得空间平移到语言这样的离散数据上，从而可以利用扩散模型思想来构建大语言模型。

**Required background（先修要求）**。鉴于本主题的技术性，我们建议具备一定的数学基础，尤其是一些概率论知识。为此我们在**附录 A** 中加入了一段简短的复习。如果里面有些概念你不熟悉，也无需担心。

## 1.3 生成式建模即采样（Generative Modeling As Sampling）

我们先思考一下可能遇到的各种数据类型，或**数据模态（data modalities）**，以及如何将它们数值化：

1. **图像（Image）**：考虑一幅具有 $H \times W$ 个像素的图像，其中 $H$ 为图像高度、$W$ 为图像宽度，每个像素有 3 个颜色通道（RGB）。对每个像素的每个颜色通道，我们都得到一个 $\mathbb{R}$ 中的强度值。因此，一幅图像可表示为一个 $z \in \mathbb{R}^{H \times W \times 3}$ 的元素。

2. **视频（Video）**：视频就是图像在时间上的序列。若有 $T$ 个时间点或**帧（frames）**，则一段视频可表示为 $z \in \mathbb{R}^{T \times H \times W \times 3}$。

3. **分子结构（Molecular structure）**：一种朴素的表示方法是把分子结构写成一个矩阵
$$
z = (z^1, \ldots, z^N) \in \mathbb{R}^{3 \times N}
$$
其中 $N$ 是分子中的原子数，每个 $z^i \in \mathbb{R}^3$ 描述该原子的位置。当然，分子还有许多更复杂的表示方式。

在以上所有例子中，我们希望生成的对象在数学上都可以表示为一个向量（必要时先展平）。因此，本文档中我们将统一写成：

> **关键思想 1（Objects as Vectors / 对象即向量）**
> 
> 我们把待生成的对象识别为向量 $z \in \mathbb{R}^d$。

一个值得注意的例外是文本数据——它通常被语言模型（如 *ChatGPT*）建模为离散对象。虽然 $z \in \mathbb{R}^d$ 的连续数据是我们的主要关注点，我们也会在**第 7 节**讨论文本生成。

**生成即采样（Generation as Sampling）**。让我们明确"生成"某样东西的含义。比如，我们想生成一张狗的图像。直观上，可接受的狗的图像有*很多*张；并不存在一张单一的"最佳"狗的图像，而是存在一个或好或坏的图像谱系。在机器学习中，我们通常用图像空间上的**概率分布（probability distribution）**来刻画这种多样性。我们把这样一个分布称为**数据分布（data distribution）**，记作 $p_{\text{data}}$。数学上，可以把 $p_{\text{data}}$ 看作一个**概率密度（probability density）**——即一个函数 $p_{\text{data}} : \mathbb{R}^d \to \mathbb{R}_{\geq 0}$，它给每个可能的 $z \in \mathbb{R}^d$ 赋予一个**似然（likelihood）**$p_{\text{data}}(z) \geq 0$。在狗图像的例子中，这个分布会对那些看起来更像狗的图像 $z$ 赋予更高的 $p_{\text{data}}(z)$。于是"图像/视频/分子有多'好'——这一相对主观的判断"就被替换为"它在数据分布 $p_{\text{data}}$ 下有多'像'（likely）"。据此，我们把"生成"任务数学化为：从（未知的）数据分布 $p_{\text{data}}$ 中采样：

> **关键思想 2（Generation as Sampling / 生成即采样）**
> 
> 生成一个对象 $z$ 被建模为从数据分布中采样 $z \sim p_{\text{data}}$。

一个**生成模型（generative model）**就是允许我们从 $p_{\text{data}}$ 中采样的机器学习模型。在机器学习中，训练模型需要数据。在生成式建模中，我们通常假设能够获得从 $p_{\text{data}}$ 中独立采样的有限多个样本，它们共同作为真实分布的代理。

> **关键思想 3（Dataset / 数据集）**
> 
> 一个数据集由有限多个样本 $z_1, \ldots, z_N \sim p_{\text{data}}$ 组成。

对图像而言，我们可以通过从互联网收集公开图像来构建数据集。对视频而言，类似地可以利用 YouTube。对蛋白质结构而言，像 RCSB Protein Data Bank (PDB) 这样的数据源提供了数十万个实验解析的结构。随着数据集规模不断增大，它将越来越接近真实分布 $p_{\text{data}}$。

**引导 / 条件生成（Guided / Conditional Generation）**。很多情况下，我们希望以某些数据 $y$ 为条件来生成对象。比如，我们想以 $y =$ "雪覆盖的山丘上奔跑的狗，背景是山脉"为条件生成图像。我们可以把它改写为从一个**条件分布（conditional distribution）**中采样：

> **关键思想 4（Guided Generation / 引导生成）**
> 
> 引导生成指从 $z \sim p_{\text{data}}(\cdot \mid y)$ 中采样，其中 $y$ 是条件变量。

我们把 $p_{\text{data}}(\cdot \mid y)$ 称为**引导数据分布（guided data distribution）**。引导生成任务通常需要学习对*任意*（而非固定的）$y$ 进行条件化。回到刚才的例子，我们可能想以不同的文本提示为条件，如 $y =$ "一张在吹生日蜡烛的猫的照片级真实感图像"。因此我们需要**同一个**模型可以对任意这样的 $y$ 进行条件化。事实证明，无条件生成的技术可以自然推广到条件情形。因此，在前 3 节里，我们将几乎只关注无条件情形（但请记住，条件生成才是我们最终的目标）。

**生成模型（Generative Models）**。抽象地说，一个生成模型就是一个返回来自 $z \sim p_{\text{data}}$ 的样本的算法（至少是近似地）。如果 $p_{\text{data}}$ 是狗的图像分布，这个算法就会返回随机的狗的图像。在本课程中，我们将聚焦于使用流或扩散模型来构建生成模型，因为它们代表了当前最先进的技术。但请记住，历史上还出现过许多其他的生成模型（未来还可能出现更多）。

> **总结 2（Generation as Sampling / 生成即采样）**
> 
> 本节要点如下：
> 
> 1. 在本工作中，我们主要考虑生成可被表示为向量 $z \in \mathbb{R}^d$ 的对象，例如图像、视频和分子结构。
> 
> 2. 生成任务即在训练时获得数据集 $z_1, \ldots, z_N \sim p_{\text{data}}$ 的前提下，从概率分布 $p_{\text{data}}$ 中采样。
> 
> 3. 引导生成假设我们对一个标签 $y$ 做条件化，希望从 $p_{\text{data}}(\cdot \mid y)$ 中采样；训练时获得的是成对数据 $(z_1, y), \ldots, (z_N, y)$。
> 
> 4. 我们的目标是构建一个生成模型——一个在训练后能够返回 $p_{\text{data}}$ 样本的模型。
