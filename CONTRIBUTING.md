# 贡献指南

感谢你愿意贡献这个翻译项目！这里给你一份简短的"翻译器上手手册"。

## 项目结构

```
MIT6.S184/                                  # 项目根 = git 仓库根
├── README.md                                # 项目主页（从这里开始）
├── LICENSE                                  # CC BY-NC-ND 4.0
├── .gitignore
├── requirements.txt                         # Python 依赖
├── CONTRIBUTING.md                          # ← 你正在看
│
├── 2506.02070v3.pdf                         # 源论文
│
├── scripts/                                 # 工具脚本（详见 scripts/README.md）
│   ├── README.md
│   ├── extract_pdf.py                       # 提取每页文本 + 渲染每页 PNG
│   ├── inspect_pdf.py                       # 查看 PDF 元数据 + 大纲
│   └── crop_*.py                            # 各章配图裁剪脚本
│
└── translations/                            # 翻译主体
    ├── README.md                            # 翻译目录说明
    ├── TRANSLATION_GUIDE.md                 # 翻译规范（必读）
    ├── ch01-introduction.md                 # 主线章节
    ├── ch02-flow-diffusion-models.md
    ├── ch03-flow-matching.md
    ├── ch04-score-functions.md
    ├── ch05-guidance.md
    ├── ch06-large-scale-generators.md
    ├── ch07-discrete-diffusion.md
    ├── appendix.md
    ├── glossary.md                          # 200+ 条术语表
    └── assets/                              # 22 张精裁配图
        ├── fig_02_01.png
        ├── fig_02_02.png
        └── ...
```

## 翻译流程（5 步）

1. **安装依赖**：`pip install -r requirements.txt`
2. **看翻译规范**：阅读 [translations/TRANSLATION_GUIDE.md](translations/TRANSLATION_GUIDE.md)，
   关键约定：术语首现中英对照、模型/产品名不译、公式严格 LaTeX
3. **查找术语**：在 [translations/glossary.md](translations/glossary.md) 查术语
   是否已收录（避免翻译漂移）
4. **校对 / 修订**：用 GitHub PR 提交小颗粒修改（建议 1-3 个公式/段为单位）
5. **更新版本号**：在 [translations/README.md](translations/README.md) 末尾的"版本"行
   标注你的修改

## 公式渲染约束

> ⚠️ **本翻译曾因 LaTeX 公式在语雀中渲染失败（显示为"非法 TeX 公式"）而修订过一次。**

请遵守以下规则，**避免在语雀 / GitHub / VSCode 预览中渲染失败**：

### ✅ 推荐写法

- 行内公式：`$x_t = \alpha_t z + \beta_t \epsilon$`
- 独立公式：`$$p_t(\cdot|z) = \mathcal{N}(\alpha_t z, \beta_t^2 I)$$`
- 多行对齐：用 `align*` 或 `aligned`，**不要在 `aligned` 中放 `\tag{}`**
- 多行公式标签：用 `^{(N)}` 上标，例如：
  ```latex
  \begin{aligned}
  X_0 &\sim p_{\text{init}}, \quad \mathrm{d}X_t = u_t^\theta(X_t)\,\mathrm{d}t} \quad (10)
  \end{aligned}
  ```

### ❌ 避免的写法

- `\begin{aligned} ... \tag{...} ... \end{aligned}` —— KaTeX 不支持，会报"非法 TeX 公式"
- `$$...$$` 之后跟 `\tag` —— 同样要避免

如果对某个公式的渲染有疑问，可以用 https://katex.org/ 在线渲染测试。

## 提交规范

- **小颗粒提交**：单次 PR 1-3 个公式 / 段
- **commit message**：参考已有 commit 风格（如果尚未初始化 git，可参考）
  - `ch02: 修订 1a/1b 公式的术语注释`
  - `glossary: 补充 "distillation" 译为"蒸馏"`
  - `typo: ch03 L154 "插值" → "中点"`
- **不要直接 force-push** 到主分支

## 不在范围内

- ❌ 重新翻译整章（除非与作者事先讨论）
- ❌ 添加大量个人解读（保持学术性）
- ❌ 修改源论文 / PDF 本身

## 联系方式

有问题先开 Issue；紧急情况可以联系仓库维护者。
