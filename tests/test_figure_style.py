"""出图脚本的样式闸门。

## 为什么值得一个测试文件

2026-10-25 修图 3 时发现一件很难看出来的事：**`axhline(..., label=...)`
配对数 y 轴会把图例排乱**。matplotlib 3.11.2 实测——条目顺序错、句柄画成
横贯整个坐标区的长线、几行文字叠在一起。图 1 的三个面板从生成那天起就是
这样，而「看一眼」完全发现不了：曲线、坐标轴、标题都对，只有图例错。

这正是本仓库第一条贡献规则要防的形态：**约束看不见地被破坏，而输出看起来
正常**。所以它不能只靠注释，必须有一个会在违规时变红的检查。

检查本身用 `ast` 而不是正则：`label=` 常常写在调用的第二行，正则要么漏
要么误伤。纯标准库，进默认 CI job。
"""

from __future__ import annotations

import ast
from pathlib import Path

SCRIPTS = sorted(Path(__file__).resolve().parent.parent.glob("scripts/make_figure*.py"))

# 这两个函数的 x/y 用的是**混合变换**（一个轴分数、一个轴数据）。带 label
# 进图例时，matplotlib 在坐标轴取对数后会排乱图例（见模块 docstring）。
BANNED = {"axhline", "axvline"}


def _labelled_span_calls(path: Path) -> list[int]:
    """返回所有 `axhline/axvline(..., label=...)` 调用的行号。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in BANNED:
            continue
        if any(kw.arg == "label" for kw in node.keywords):
            hits.append(node.lineno)
    return hits


def test_figure_scripts_exist():
    """闸门本身要有负控：一个脚本都没扫到 = 检查被静默跳过。"""
    assert SCRIPTS, "没找到 scripts/make_figure*.py， glob 路径写错了"


def test_no_labelled_axhline_or_axvline():
    """带 label 的 axhline/axvline 一律禁止。

    替代写法：`ax.plot([0, 1], [y, y], transform=ax.get_xaxis_transform(),
    label=...)`。横向上仍然满宽，但图例里的句柄是正常的一小段。
    """
    for path in SCRIPTS:
        hits = _labelled_span_calls(path)
        assert not hits, (
            f"{path.name} 第 {hits} 行给 axhline/axvline 传了 label。"
            "对数坐标轴上这会把图例排乱（matplotlib 3.11.2 实测），"
            "改用 ax.plot(..., transform=ax.get_xaxis_transform(), label=...)"
        )


def test_the_guard_actually_detects_a_violation():
    """闸门必须有负控：永远 PASS 的检查比没有检查更糟。"""
    bad = "import matplotlib.pyplot as plt\nax.axhline(1.0, label='x')\n"
    tree = ast.parse(bad)
    hits = [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in BANNED
            and any(kw.arg == "label" for kw in n.keywords)]
    assert hits == [2]
