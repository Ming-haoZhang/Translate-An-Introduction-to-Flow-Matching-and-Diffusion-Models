"""
Fix 'illegal TeX formula' in Yuque (and GitHub) by removing \\tag{...} from
\\begin{aligned} ... \\end{aligned} blocks.

Why: KaTeX (which Yuque uses) does NOT support \\tag inside aligned.
    MathJax supports it, but the spec is that \\tag belongs in
    equation/align/gather, not aligned (which is an *embedded* env).

Strategy: For every aligned block in any .md file under translations/,
    replace '\\tag{N}' with '^{(N)}' (text label in superscript).
    This preserves the meaning ("this is equation (N)") while
    making the block render in KaTeX.

Run:  python scripts/fix_aligned_tags.py
"""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD_FILES = sorted((ROOT / "translations").glob("*.md"))

# Match: \begin{aligned} ... \end{aligned}  (non-greedy, DOTALL)
ALIGNED_RE = re.compile(
    r"\\begin\{aligned\}(?P<body>.*?)\\end\{aligned\}",
    re.DOTALL,
)
# Match \tag{...} inside the block
TAG_RE = re.compile(r"\\tag\{(?P<num>[^}]*)\}")


def fix_aligned(match: re.Match) -> str:
    body = match.group("body")
    new_body = TAG_RE.sub(r"^{ (\g<num>) }", body)
    return f"\\begin{{aligned}}{new_body}\\end{{aligned}}"


def main() -> None:
    total_blocks = 0
    total_tags = 0
    for f in MD_FILES:
        text = f.read_text(encoding="utf-8")
        blocks = list(ALIGNED_RE.finditer(text))
        if not blocks:
            continue
        n_blocks = len(blocks)
        n_tags = sum(len(TAG_RE.findall(b.group("body"))) for b in blocks)
        if n_tags == 0:
            continue
        new_text = ALIGNED_RE.sub(fix_aligned, text)
        # Backup
        (f.with_suffix(f".md.bak")).write_text(text, encoding="utf-8")
        f.write_text(new_text, encoding="utf-8")
        total_blocks += n_blocks
        total_tags += n_tags
        print(f"  {f.name}: fixed {n_tags} \\tag in {n_blocks} aligned block(s)")

    print(f"\nDone. {total_tags} \\tag removed across {total_blocks} aligned blocks.")
    print("Backups saved as <name>.md.bak — delete them after verification.")


if __name__ == "__main__":
    main()
