"""`flsoh.stats` 的测试。

为什么这层必须钉死：H1–H3 的 accept/reject 全部由这里的 p 值决定。
p 值算错不会让任何东西报错——它只会让论文里写「推翻」的地方其实该写
「未判定」。所以每个函数都要对已知的教科书值验一遍。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

from flsoh.stats import (
    _binom_two_sided,
    is_significant,
    paired_stats,
    t_two_sided_p,
)


class TestTDistribution:
    """对 t 分布表的临界值：t_{0.025, df} 处双侧 p 必须是 0.05。"""

    @pytest.mark.parametrize(
        "t,df",
        [
            (4.303, 2),
            (3.182, 3),
            (2.776, 4),
            (2.571, 5),
            (2.447, 6),
            (2.306, 8),
            (2.262, 9),
            (2.228, 10),
            (2.086, 20),
            (2.042, 30),
        ],
    )
    def test_critical_values(self, t, df):
        assert t_two_sided_p(t, df) == pytest.approx(0.05, abs=2e-3)

    def test_zero_t_gives_p_one(self):
        assert t_two_sided_p(0.0, 7) == pytest.approx(1.0, abs=1e-12)

    def test_symmetric(self):
        assert t_two_sided_p(-3.0, 5) == pytest.approx(t_two_sided_p(3.0, 5))

    def test_large_t_goes_to_zero(self):
        assert t_two_sided_p(1e6, 3) < 1e-10

    def test_infinite_t(self):
        assert t_two_sided_p(math.inf, 4) == 0.0

    def test_rejects_bad_df(self):
        with pytest.raises(ValueError):
            t_two_sided_p(1.0, 0)


class TestSignTest:
    def test_three_seeds_can_never_be_significant(self):
        """n=3 时最小双侧 p 是 2/8=0.25——这就是「3 seeds 判不出来」的数学原因。"""
        assert _binom_two_sided(0, 3) == pytest.approx(0.25)
        assert _binom_two_sided(3, 3) == pytest.approx(0.25)

    def test_six_seeds_all_same_way(self):
        assert _binom_two_sided(0, 6) == pytest.approx(2 / 64)

    def test_eight_seeds_all_same_way(self):
        assert _binom_two_sided(0, 8) == pytest.approx(2 / 256)

    def test_split_is_not_significant(self):
        assert _binom_two_sided(2, 4) == pytest.approx(1.0)

    def test_one_against_seven(self):
        # k=1, n=8: 2*(C(8,0)+C(8,1))/256 = 2*9/256
        assert _binom_two_sided(1, 8) == pytest.approx(18 / 256)


class TestPairedStats:
    def test_hand_computed(self):
        """手算：diffs = [-1, -2, -3]，mean=-2, sd=1, se=1/sqrt(3)。"""
        res = paired_stats([-1.0, -2.0, -3.0])
        assert res["n"] == 3
        assert res["mean"] == pytest.approx(-2.0)
        assert res["sd"] == pytest.approx(1.0)
        assert res["se"] == pytest.approx(1 / math.sqrt(3))
        assert res["t"] == pytest.approx(-2.0 * math.sqrt(3))
        assert res["df"] == 2
        assert res["n_neg"] == 3
        assert res["n_pos"] == 0

    def test_all_zero_diffs(self):
        """差恒为零时 t=0、p=1，不能除零。"""
        res = paired_stats([0.0, 0.0, 0.0])
        assert res["sd"] == 0.0
        assert res["t"] == 0.0
        assert res["p_t"] == pytest.approx(1.0)

    def test_constant_nonzero_diffs_give_infinite_t(self):
        res = paired_stats([0.5, 0.5, 0.5])
        assert math.isinf(res["t"])
        assert res["p_t"] == 0.0

    def test_needs_two_seeds(self):
        with pytest.raises(ValueError):
            paired_stats([1.0])

    def test_ci_brackets_mean(self):
        res = paired_stats([-1.0, -2.0, -3.0, -2.5])
        assert res["ci_low"] < res["mean"] < res["ci_high"]


class TestIsSignificant:
    def _res(self, **kw):
        base = {
            "n": 3, "mean": 0.0, "sd": 1.0, "se": 1.0, "t": 0.0, "df": 2,
            "p_t": 1.0, "p_sign": 1.0, "n_neg": 0, "n_pos": 0,
            "ci_low": 0.0, "ci_high": 0.0,
        }
        base.update(kw)
        return base

    def test_not_significant_when_p_large(self):
        sig, why = is_significant(self._res(p_t=0.4))
        assert not sig
        assert "p_t" in why

    def test_significant_via_t_even_if_sign_test_fails(self):
        """n=3 时符号检验必然失败，但 t 可以显著——不能因此判「未判定」。"""
        sig, why = is_significant(self._res(p_t=0.01, p_sign=0.25))
        assert sig
        assert "符号检验" in why  # 必须把这个弱点写出来，不能掩盖

    def test_significant_and_robust(self):
        sig, why = is_significant(self._res(n=8, p_t=0.001, p_sign=0.008))
        assert sig
        assert "符号检验" in why
