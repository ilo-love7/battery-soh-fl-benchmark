"""README 的测试数徽章必须与仓库里的真实测试数一致。

为什么值得单独测
----------------
徽章数字是**最容易过期**的那种数字：它不参与任何计算，没有任何测试会
因为「数字对不上」而红，于是它会在某次提交里悄悄变成假的。这个仓库的
第一条贡献规则是「每条硬约束都要有可执行检查」——徽章声明了测试数，
这就是一条硬约束。

环境无关是关键：``pytest`` 收集到的数量随「装没装 numpy / flwr / opacus」
而变（裸 Python 297，全依赖 493），拿任何一个当基准都会在另一个环境里
误报。所以这里用 ``ast`` **静态数测试函数**——数与装了哪些包无关，
在只装 pytest 的默认 CI job 里同样能跑。

同时盯中英双版：两份 README 的章节一一对应，徽章漂移通常只改一份。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = REPO_ROOT / "tests"

#: 与 README 里的 shields.io 徽章一致：tests-<N>
_BADGE_RE = re.compile(r"badge/tests-(\d+)")


def count_test_functions(test_dir: Path = TEST_DIR) -> int:
    """静态数出仓库里定义的测试函数个数（不含 parametrize 展开）。"""
    total = 0
    for path in sorted(test_dir.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test"):
                    total += 1
    return total


def _badge_numbers(text: str) -> list[int]:
    return [int(m) for m in _BADGE_RE.findall(text)]


class TestBadgeMatchesReality:
    def test_badge_exists_in_both_readmes(self):
        """双版都必须有徽章，且形状对得上。"""
        for name in ("README.md", "README.zh-CN.md"):
            text = (REPO_ROOT / name).read_text(encoding="utf-8")
            numbers = _badge_numbers(text)
            assert numbers, f"{name} 里找不到 tests-<N> 徽章"
            assert len(set(numbers)) == 1, f"{name} 的徽章数字不一致：{numbers}"

    def test_badge_equals_static_count(self):
        want = count_test_functions()
        assert want > 0, "一个测试函数都没数到——glob 或 AST 逻辑坏了"
        for name in ("README.md", "README.zh-CN.md"):
            text = (REPO_ROOT / name).read_text(encoding="utf-8")
            got = _badge_numbers(text)[0]
            assert got == want, (
                f"{name} 的徽章写着 {got}，仓库里实际有 {want} 个测试函数。"
                "改徽章之前先确认新测试真的写完了——这条检查拦的就是"
                "「README 声称有测试、实际没写」"
            )

    def test_badge_number_is_not_a_stale_round_number(self):
        """拦「随手写个整数」：徽章数必须能被静态清点复现，不能是估的。

        这一条是上一条的负控：上一条注入乱值，这里注入一个**看起来很合理
        的旧值**（真值 −1）。提取逻辑必须真的把这个数读出来，否则上一条
        只是在跟一个永远匹配的空字符串比。
        """
        text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        got = _badge_numbers(text)[0]
        real = count_test_functions()
        assert got == real
        # 把徽章换成一个差 1 的旧值，提取逻辑必须读出那个旧值
        stale_text = text.replace(f"badge/tests-{got}", f"badge/tests-{got - 1}")
        assert _badge_numbers(stale_text) == [got - 1]
        assert got - 1 != real


class TestStaticCounter:
    """静态计数器自己也要测：它算错的话，上面的闸门会忠实地把错的数固化。"""

    def test_counts_a_known_fixture(self, tmp_path):
        (tmp_path / "test_a.py").write_text(
            "def test_one():\n    pass\n\n\n"
            "class TestX:\n    def test_two(self):\n        pass\n\n"
            "def helper():\n    pass\n\n"
            "def test_three():\n    pass\n",
            encoding="utf-8",
        )
        assert count_test_functions(tmp_path) == 3

    def test_ignores_non_test_files(self, tmp_path):
        (tmp_path / "helper.py").write_text("def test_nope():\n    pass\n",
                                           encoding="utf-8")
        assert count_test_functions(tmp_path) == 0

    def test_empty_dir_counts_zero(self, tmp_path):
        assert count_test_functions(tmp_path) == 0
