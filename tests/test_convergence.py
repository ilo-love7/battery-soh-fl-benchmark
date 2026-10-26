"""收敛闸门的测试。

这个检查器的输出会直接决定「某个家族能不能参与判定」，所以它自己必须
先被钉住。尤其是两条：**尖峰不许把结论带走**（稳健统计的负控），以及
**窗口按比例取**（否则 C0 与联邦被两把尺子量）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import check_convergence as cc


class TestWindow:
    def test_scales_with_curve_length(self):
        # C0：60 个点（每 5 epoch）→ 12 点 = 末 60 epoch
        assert cc.window_for(60) == 12
        # 联邦：100 轮 → 20 轮
        assert cc.window_for(100) == 20

    def test_has_a_floor(self):
        assert cc.window_for(20) == 10

    def test_fixed_point_count_would_measure_different_spans(self):
        """负控：固定 20 点会让 C0 的窗口覆盖三分之一个训练过程。"""
        assert cc.window_for(60, frac=0.2) != 20
        assert 20 / 60 > 0.3  # 固定 20 点在 60 点曲线上占 33%


class TestRobustSlope:
    def test_flat_series_has_zero_slope(self):
        assert cc.robust_slope([0.001] * 20) == pytest.approx(0.0)

    def test_descending_series_is_negative(self):
        # 每步降 1% → 半窗（10 步）中位差 −10%，每步 −1%
        tail = [0.001 * (0.99 ** i) for i in range(20)]
        assert cc.robust_slope(tail) == pytest.approx(-0.01, abs=2e-3)

    def test_single_spike_at_the_end_does_not_flip_the_sign(self):
        """负控：C0 的轨迹就是这种形态（seed2 末段冲到 0.0249）。

        最小二乘会被末端一个 10 倍离群点拉成**正斜率**（"在变差"），
        而半窗中位差仍读到原来的轻微下降趋势。
        """
        tail = [0.0020] * 19 + [0.0200]
        n = len(tail)
        xs = list(range(n))
        mx = sum(xs) / n
        my = sum(tail) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, tail))
        den = sum((x - mx) ** 2 for x in xs)
        ls_slope = (num / den) / my
        assert ls_slope > 0          # 最小二乘被骗成正斜率
        assert cc.robust_slope(tail) == pytest.approx(0.0, abs=1e-9)

    def test_too_few_points_is_zero(self):
        assert cc.robust_slope([0.001, 0.002]) == 0.0


class TestDiagnostics:
    def test_flat_curve_converges(self):
        d = cc.diagnostics([0.002] * 60)
        assert d["converged"] is True
        assert d["reason"] == ""

    def test_still_descending_does_not_converge(self):
        d = cc.diagnostics([0.01 * (0.98 ** i) for i in range(100)])
        assert d["converged"] is False
        assert "仍在降" in d["reason"]

    def test_jittery_but_flat_does_not_converge(self):
        # 中位不变，但四分位间距大：平台附近乱走，`final_mae` 是抽签
        maes = ([0.001] * 50) + ([0.0005, 0.0020] * 25)
        d = cc.diagnostics(maes)
        assert d["converged"] is False
        assert "IQR" in d["reason"]

    def test_short_curve_does_not_converge(self):
        d = cc.diagnostics([0.001, 0.002, 0.003])
        assert d["converged"] is False
        assert "不足" in d["reason"]

    def test_naive_spread_would_flag_a_single_spike(self):
        """负控：用 (max−min)/mean 时，一个尖峰就把 IQR 很小的曲线判成未收敛。

        实测 C0 的朴素抖动 94.7%、稳健 IQR 27.3%——那个 94.7% 只说明
        「有一个坏点」，不说明「没收敛」。
        """
        maes = ([0.002] * 59) + [0.025]
        naive = (max(maes) - min(maes)) / (sum(maes) / len(maes))
        assert naive > 1.0
        d = cc.diagnostics(maes)
        assert d["spread"] < 0.2


class TestBuildReport:
    def _run(self, run_id, maes):
        return {
            "run_id": run_id,
            "final_mae": maes[-1],
            "curve": [{"round": i + 1, "mae": m} for i, m in enumerate(maes)],
            "_maes": maes,
        }

    def test_groups_by_family_and_counts_converged(self):
        runs = [
            self._run("x__seed0", [0.002] * 60),
            self._run("x__seed1", [0.01 * (0.98 ** i) for i in range(60)]),
            self._run("y__seed0", [0.003] * 60),
        ]
        rows = {r["family"]: r for r in cc.build_report(runs, 0.2)}
        assert rows["x"]["n"] == 2 and rows["x"]["converged"] == 1
        assert rows["y"]["converged"] == 1

    def test_runs_without_curve_are_skipped(self):
        runs = [{"run_id": "z__seed0", "final_mae": 0.1, "curve": [], "_maes": []}]
        assert cc.build_report(runs, 0.2) == []


class TestFamilyOf:
    def test_strips_seed_suffix(self):
        assert cc.family_of("fedavg__iid__seed3") == "fedavg__iid"
        assert cc.family_of("fedprox-mu0.01-dirichlet0.5-seed7") == "fedprox-mu0.01-dirichlet0.5"

    def test_id_starting_with_seed_is_not_emptied(self):
        assert cc.family_of("seed3") == "seed3"
