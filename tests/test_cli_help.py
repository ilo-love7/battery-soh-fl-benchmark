"""argparse help 字符串的端口性闸门。

## 为什么值得一个测试文件

argparse 会拿 help 字符串做 `%` 格式化。**帮助文本里出现一个半角 `%` 就是
一颗定时炸弹**，而它在 Python 3.11–3.13 上完全不响：那些版本只在字符串里
出现 `%(` 时才格式化，所以 `"占墙钟 98.2%，看着像"` 一直显示得好好的。

到 Python 3.14 换了行为，同一行直接

    ValueError: unsupported format character '?' (0xff0c) at index 58

—— `--help` 崩、任何解析参数的程序都崩，而报错指向 argparse 内部，看不出
是帮助文本写错了。本仓库 `scripts/build_features.py` 的 `--workers` 就踩了
一次（`98.2%，` 的半角 % 后面跟着全角逗号）。

修法有两种，都不完美：

- 写成 `%%`：3.14 上显示成一个 `%`，但 ≤3.13 不格式化，会显示成 `%%`。
- 用**全角 ％（U+FF05）**：任何版本都原样显示，且中文排版里全角百分号
  本来就比半角合适。本仓库采用这一种。

这个文件用 `ast` 扫所有脚本，**不 import 它们**（脚本顶部有重依赖），所以
能进默认 CI job。
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = (sorted(ROOT.glob("scripts/*.py"))
           + sorted(ROOT.glob("src/flsoh/*.py"))
           + sorted(ROOT.glob("src/flsoh/app/*.py")))

FULLWIDTH_PERCENT = "\uff05"


def _help_strings(path: Path) -> list[tuple[int, str]]:
    """返回该文件所有 `add_argument(..., help="...")` 的 (行号, 文本)。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue
        for kw in node.keywords:
            if kw.arg == "help" and isinstance(kw.value, ast.Constant):
                if isinstance(kw.value.value, str):
                    out.append((node.lineno, kw.value.value))
    return out


def test_targets_exist():
    """闸门要有负控：一个文件都没扫到 = 检查被静默跳过。"""
    assert len(TARGETS) >= 10, f"只扫到 {len(TARGETS)} 个文件，glob 路径写错了"


def test_no_bare_percent_in_help_strings():
    """help 文本里不许有半角 `%`。

    全角 ％ 不受影响。`%%` 也不许——它在 ≤3.13 上会显示成两个百分号，
    属于「在一个 Python 版本上正确的修法」。
    """
    bad = []
    for path in TARGETS:
        for lineno, text in _help_strings(path):
            if "%" in text:
                bad.append(f"{path.name}:{lineno}: {text[:60]!r}")
    assert not bad, (
        "argparse 的 help 文本里有半角 %，Python 3.14 上会让 --help 直接崩"
        "（argparse 会对其做 % 格式化）：\n  " + "\n  ".join(bad)
        + f"\n  改用全角 {FULLWIDTH_PERCENT}。"
    )


def test_the_guard_actually_detects_a_violation():
    """负控：注入一个坏字符串，闸门必须报出来。"""
    src = "import argparse\np.add_argument('--x', help='98.2%，看着像')\n"
    tree = ast.parse(src)
    hits = [n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "add_argument"
            and any(kw.arg == "help" and "%" in getattr(kw.value, "value", "")
                    for kw in n.keywords)]
    assert len(hits) == 1
