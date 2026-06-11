# MIT 6.S184 流匹配与扩散模型 — 中文翻译术语表

> 本文档是 *An Introduction to Flow Matching and Diffusion Models*（arXiv:2506.02070v3，Peter Holderrieth & Ezra Erives，CC BY-NC-ND 4.0，84 页）中文翻译的**统一术语表**。所有 8 章翻译中术语用法保持一致；首次出现时**中文（English）**，后续只用中文（模型/产品/方法名按"不译"原则保留英文）。

---

## 1. 数学基础（Mathematical Foundations）

| 英文 | 中文 | 备注 |
|---|---|---|
| random variable (RV) | 随机变量 | |
| probability density function (PDF) | 概率密度函数 | |
| probability path | 概率路径 | 描述 $p_t$ 随 $t$ 演化的轨迹 |
| conditional probability path | 条件概率路径 | $p_t(\cdot \mid x_1)$ |
| marginal probability path | 边缘概率路径 | $p_t(x) = \mathbb{E}_{x_1 \sim p_{\text{data}}}[p_t(x \mid x_1)]$ |
| data distribution | 数据分布 | $p_{\text{data}}$ |
| prior / initial distribution | 先验 / 初始分布 | $p_{\text{init}}$ 或 $p_{\text{prior}}$ |
| prior distribution | 先验分布 | 潜变量 $z$ 的分布 $p_{\text{prior}}$ |
| expectation | 期望 | $\mathbb{E}[\cdot]$ |
| conditional expectation | 条件期望 | $\mathbb{E}[X \mid Y]$ |
| variance | 方差 | |
| covariance matrix | 协方差矩阵 | |
| inner product | 内积 | $\langle x, y \rangle$ |
| norm | 范数 | $\|x\|$ |
| gradient | 梯度 | $\nabla f$ |
| Jacobian | 雅可比 | |
| divergence | 散度 | $\mathrm{div}$ |
| Laplacian | 拉普拉斯算子 | $\Delta$ |
| Hessian | 海森矩阵 | $\nabla^2$ |
| ordinary differential equation (ODE) | 常微分方程 | |
| stochastic differential equation (SDE) | 随机微分方程 | |
| Brownian motion / Wiener process | 布朗运动 / 维纳过程 | $W_t$ |
| Ornstein-Uhlenbeck (OU) process | Ornstein–Uhlenbeck 过程 | $\mathrm{d}X = -\theta X \mathrm{d}t + \sigma \mathrm{d}W$ |
| diffusion coefficient | 扩散系数 | |
| noise schedule | 噪声调度 | $\sigma_t$、$\beta_t$ |
| continuous-time Markov chain (CTMC) | 连续时间马尔可夫链 | 离散数据的类比 |
| rate matrix / generator | 比率矩阵 / 生成元 | $Q_t(y \mid x)$ |
| transition kernel | 转移核 | $p_{t' \mid t}$ |
| Markov property | 马氏性 | |
| Chapman-Kolmogorov equation | 查普曼-科尔莫戈罗夫方程 | |
| continuous-time Fokker-Planck equation | 连续时间 Fokker-Planck 方程 | (定理 11 / 41) |
| continuity equation | 连续性方程 | 边缘密度演化（$\sigma = 0$ 情形） |
| Kolmogorov forward equation (KFE) | Kolmogorov 前向方程 | 离散情形的 (命题 2) |
| Euler method | Euler 方法 | |
| Euler-Maruyama method | Euler–丸山方法 | SDE 离散化 |
| Heun's method / Heun's second-order method | Heun 方法 / 二阶 Heun 方法 | |
| Picard theorem | Picard 定理 | ODE 存在唯一性 |
| test function | 测试函数 | 光滑 + 紧支集 |

---

## 2. 概率机器学习（Probabilistic ML）

| 英文 | 中文 | 备注 |
|---|---|---|
| generative model | 生成模型 | |
| flow model | 流模型 | 由 ODE 驱动的生成模型 |
| diffusion model | 扩散模型 | 由 SDE 驱动的生成模型 |
| denoising diffusion probabilistic model (DDPM) | 去噪扩散概率模型 | |
| flow matching | 流匹配 | 训练流模型的方法 |
| conditional flow matching (CFM) | 条件流匹配 | 训练算法（Lipman et al. 2022） |
| rectified flow | 整流流 | 直线插值路径（Liu et al. 2022） |
| score function | 分数函数 | $\nabla \log p_t$ |
| score matching | 分数匹配 | 训练 $s_\theta \approx \nabla \log p$ 的方法 |
| denoising score matching | 去噪分数匹配 | 训练目标（Hyvärinen 2005 / Song et al. 2021） |
| denoiser | 去噪器 | $\epsilon_\theta$ / $x_\theta$ |
| conditional score | 条件分数 | $\nabla \log p_t(x \mid x_1)$ |
| marginal score | 边缘分数 | $\nabla \log p_t(x)$ |
| noise prediction / $\epsilon$-prediction | 噪声预测 / $\epsilon$ 预测 | $\epsilon_\theta(x_t, t) \approx \epsilon$ |
| Langevin dynamics | 朗之万动力学 | $\mathrm{d}x = \nabla \log p(x)\, \mathrm{d}t + \sqrt{2}\, \mathrm{d}W$ |
| probability flow ODE | 概率流 ODE | 与某 SDE 边缘分布一致的确定性 ODE |
| stochastic interpolant (SI) | 随机插值 | [1] Albergo et al. 2023 |
| vector field | 向量场 | $u_t$ |
| conditional vector field | 条件向量场 | $u_t(\cdot \mid x_1)$ |
| marginal vector field | 边缘向量场 | $u_t$（学习目标） |
| flow | 流 | $\psi_t : \mathbb{R}^d \to \mathbb{R}^d$ |
| diffeomorphism | 微分同胚 | |
| unconditional / guided / conditional | 无条件 / 引导 / 条件 | 区分无 $y$ / 显式 $y$ 监督 / 从联合采样 |
| classifier-free guidance (CFG) | 无分类器引导 | |
| classifier guidance | 分类器引导 | |
| guidance scale / weight | 引导强度 / 引导权重 | $w$ |
| empty label | 空标签 | $\varnothing$ |
| guidance variable | 引导变量 | $y$ |
| interpolant | 插值 | $I(t, x, z)$ |
| mean / variance-preserving (VP/VE/...) | 均值 / 方差保持型 | SDE 设计类型 |

---

## 3. 离散扩散（Discrete Diffusion, Ch.7）

| 英文 | 中文 | 备注 |
|---|---|---|
| discrete diffusion | 离散扩散 | 离散状态空间的扩散 |
| state space | 状态空间 | $S = V^d$ |
| vocabulary | 词汇表 | $V$ |
| one-hot vector | 单热向量 | |
| simplex | 单纯形 | $\Delta^K$ |
| categorical distribution | 类别分布 | $\text{Categorical}$ |
| absorbing state | 吸收状态 | 一旦进入就不再变化的 token（如 [MASK]） |
| masking path | 掩码路径 | 离散条件路径的"自然"选择 |
| rate matrix | 比率矩阵 | $Q_t(y \mid x)$ |
| marginal rate matrix | 边缘比率矩阵 | 条件比率的后验加权平均 |
| conditional rate matrix | 条件比率矩阵 | $R_t(b \mid a, x_1)$ |
| ancestral sampling | 祖先采样 | 离散时间 CTMC 采样 |
| non-autoregressive | 非自回归 | 区别于 GPT 逐 token 生成 |
| masking diffusion language model (MDLM) | 掩码扩散语言模型 | $p_{\text{init}} = \delta_{\text{[MASK]}}$ 的离散扩散 |
| factorized model | 因子化模型 | 逐 token 独立的 CTMC |
| likelihood-based discrete flow matching | 基于似然的离散流匹配 | |
| D3PM | D3PM | [5] Campbell et al. 2022 |
| GLIDE / SEDD | 略 | 离散扩散特定模型 |

---

## 4. 神经网络架构（Neural Network Architectures, Ch.6）

| 英文 | 中文 | 备注 |
|---|---|---|
| neural network (NN) | 神经网络 | |
| multi-layer perceptron (MLP) | 多层感知机 | |
| U-Net | U-Net | 不译 |
| Transformer | Transformer | 不译 |
| diffusion Transformer (DiT) | 扩散 Transformer | |
| multimodal DiT (MM-DiT) | 多模态扩散 Transformer | Stable Diffusion 3 架构 |
| DiT block | DiT 块 | |
| modulation / adaptive normalization (AdaLN) | 调节 / 自适应归一化 | |
| AdaLN-Zero | AdaLN-Zero | 初始化为 0 的 AdaLN |
| patch embedding | 块嵌入 | |
| patchify | 块化 | |
| positional encoding | 位置编码 | |
| self-attention | 自注意力 | |
| cross-attention | 交叉注意力 | |
| multi-head self-attention | 多头自注意力 | |
| scaled dot-product attention | 缩放点积注意力 | |
| scaled cosine attention | 缩放余弦注意力 | Stable Diffusion 3 用 |
| FiLM layer | FiLM 层 | 特征线性调制 |
| downsampling / upsampling | 下采样 / 上采样 | |
| skip connection | 跳跃连接 | |
| residual block | 残差块 | |
| VEO-3 / SORA | VEO-3 / SORA | 不译（Google / OpenAI 视频模型） |
| Movie Gen | Movie Gen | Meta 视频模型 |
| Nano Banana | Nano Banana | 图像模型（保留） |
| FLUX 2.0 | FLUX 2.0 | Black Forest Labs 图像模型 |
| Stable Diffusion 3 | Stable Diffusion 3 | Stability AI 图像模型 |
| LLaDA | LLaDA | 大语言扩散模型 |
| CLIP | CLIP | OpenAI 图像-文本对齐 |
| T5 | T5 | 文本编码器 |
| BYT5 | ByT5 | 字节级 T5 |
| video temporal autoencoder (TAE) | 视频时间自编码器 | Movie Gen 用 |
| temporal tiling | 时间平铺 | 视频降采样策略 |
| 3D U-Net | 3D U-Net | 视频生成架构 |
| midcoder | 中间编码器 | Movie Gen 用 |
| U-ViT | U-ViT | Transformer + U-Net 混合 |

---

## 5. 潜空间与变分自编码器（Latent Space & VAE, Ch.6 + App.D）

| 英文 | 中文 | 备注 |
|---|---|---|
| latent diffusion model (LDM) | 潜在扩散模型 | 在 VAE 潜空间做扩散 |
| autoencoder (AE) | 自编码器 | |
| variational autoencoder (VAE) | 变分自编码器 | |
| encoder | 编码器 | $q_\phi(z \mid x)$ |
| decoder | 解码器 | $p_\theta(x \mid z)$ |
| latent code | 潜变量 | $z$ |
| latent space | 潜空间 | |
| reparameterization trick | 重参数化技巧 | |
| KL divergence | KL 散度 | $D_{\mathrm{KL}}$ |
| KL warm-up | KL 预热 | |
| posterior collapse | 后验坍缩 | |
| reconstruction loss | 重建损失 | |
| perceptual loss | 感知损失 | |
| adversarial loss | 对抗损失 | |
| evidence lower bound (ELBO) | 证据下界 | |
| amortized inference | 摊销推断 | |
| data-processing inequality | 数据处理不等式 | |
| rate | 速率 | 信息论中指"每维 bits" |
| distortion | 失真 | 重建质量（RMSE 等） |
| reconstruction-FID (rFID) | 重建 FID | |
| generative-FID (gFID) | 生成 FID | |
| Frechet inception distance (FID) | Frechet Inception 距离 | |
| amortization gap | 摊销间隙 | VAE 解码器与潜空间先验的失配 |
| downsampling factor | 下采样因子 | $f$ |
| latent channel dimension | 潜变量通道维度 | $d$ |
| Pareto frontier | 帕累托前沿 | rate-distortion 曲线 |

---

## 6. 训练与采样（Training & Sampling）

| 英文 | 中文 | 备注 |
|---|---|---|
| training / training iteration | 训练 / 训练迭代 | |
| stochastic gradient descent (SGD) | 随机梯度下降 | |
| Adam / AdamW | Adam / AdamW | 优化器 |
| optimizer step | 优化器步 | |
| mini-batch / batch | 小批 / 批 | |
| overfitting | 过拟合 | |
| generalization | 泛化 | |
| FID / IS / LPIPS | FID / IS / LPIPS | 评估指标 |
| classifier-free guidance (CFG) | 无分类器引导 | 评估时常用 |
| inference | 推理 | |
| sampling | 采样 | |
| Euler step | Euler 步 | |
| Heun step | Heun 步 | |
| NFE | 神经函数求值次数 | number of function evaluations |
| noise schedule | 噪声调度 | |
| conditioning | 条件化 | |
| classifier | 分类器 | |
| training target | 训练目标 | $u_t^{\text{target}}$ |
| flow matching loss | 流匹配损失 | |
| conditional flow matching loss | 条件流匹配损失 | CFM |
| denoising loss | 去噪损失 | |
| cross-entropy loss | 交叉熵损失 | |
| log-likelihood | 对数似然 | |
| negative log-likelihood (NLL) | 负对数似然 | |
| DFM loss (discrete flow matching) | DFM 损失 | $\mathcal{L}_{\text{DFM}}$ |
| masking language modeling (MLM) | 掩码语言模型 | |
| exponential moving average (EMA) | 指数移动平均 | |

---

## 7. 关键模型 / 产品名（Do NOT translate）

以下保留英文原文：

- **公司 / 实验室**：Google, OpenAI, Meta, Stability AI, Black Forest Labs, Anthropic, NVIDIA, Apple, Microsoft
- **产品 / 模型**：Stable Diffusion 1/2/3、SD3、SDXL、SD3.5、FLUX、FLUX 2.0、Nano Banana、VEO-3、SORA、Movie Gen、AlphaFold3、ChatGPT、GPT-4、Imagen、DALL·E、Midjourney、eDiff-I
- **学术名词 / 算法名**（在主线中已给中文，但首次出现时建议同时给英文）：flow matching、score matching、DDPM、CTMC、ODE、SDE、Heun's method、Euler-Maruyama、Fokker-Planck、CondOT、MaskGIT、MDLM、MD4、DUO、Masked Diffusion、Rectified Flow
- **架构 / 方法名**：U-Net、Transformer、DiT、MM-DiT、CLIP、VAE、LDM、ELBO、AdaLN、AdaLN-Zero、FiLM、ReLU、GELU、SiLU
- **常见变量符号**（保留 $X_t$、$u_t$、$p_t$ 等原符号，不译为中文变量）

---

## 8. 不译原则（Do NOT Translate Rule）

- 引用编号：保留 `[1]`、`[17]` 等原始编号（已在 References 中给出完整条目）。
- 论文标题：保留英文原标题。
- 作者名：保留英文姓名（首次出现可加中文译名但建议不译）。
- 章节标题中的副标题：用 `/` 分隔中英文（如 `离散扩散（Discrete Diffusion）`）。
- 章节 / 定理 / 公式编号：保留 `Theorem 11`、`Eq 87`、`Algorithm 7` 等原编号。
- 章节中直接引用的英文原句：保留英文原文（特别是关键引用、Term、原论文标题）。

---

**关于术语用法约定**：
1. **首现原则**：术语**第一次出现**时写「中文（English）」，后续只用中文。例外：模型/产品/方法名/变量符号始终保留英文。
2. **同义异译**：禁止"分数函数"和"得分函数"混用——已统一为「分数函数」。
3. **专有名词首字母大小写**：Diffusion Model、Flow Matching、Score Function、Continuous-Time Markov Chain 等在英文原文中**大写**的术语，译为中文后**不**在句首时也无需大写（中文无大小写概念）。
4. **公式变量**：$X_t$、$p_t$、$u_t^\theta$、$Q_t$、$R_t$ 等公式变量**保留原文符号**——不要改为"$\text{状态}_t$"、"$\text{概率}_t$"等。

---

> **版本**：v3（2026-06-10）
> **覆盖范围**：8 个主章节（ch01–ch07）+ 附录（p.65–84）
> **总翻译量**：~200 KB md + 22 张精裁配图
