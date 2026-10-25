"""分段与特征健康检查（2026-10-25 抓到的那个 bug 的守门测试）。

为什么值得单独测
----------------
2026-10-25 发现：``features.parquet`` 里那 2700 行「ΔQ(V)」是把**整条循环**
（充电 + 放电拼在一起）喂给 ``delta_q_curve`` 算出来的。它的 ``argsort`` 于是
把放电段排进充电段的电压区间，而放电段的 ``Qc`` 是常数——低电压区冒出一片
「容量 = 满容量」的平台。

三个证据（全部用 ``tests/fixtures/`` 下的真实 MATR 数据复现）：

1. 总变差：整条循环 **269.60**，仅放电段 **0.21**，差 1280 倍；
2. 两条曲线的相关系数 **−0.019**——基本不相干；
3. 而那 2700 行照样训出了 MAE 0.0016 的模型，精度曲线完全正常。

所以这不是「精度差一点」，是**两套不相干的数据**。而它能藏住，是因为没有任何
检查看过特征的**光滑性**。这个文件就是那道检查。

夹具是从 MATR 批次 1 的 b1c0 真实抓下来的循环 1 与循环 5（各 1087/1102 点），
不是合成的——合成数据测不出「两段拼接」这个结构，只有同构的真实数据能。
"""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="分段需要 numpy")
pytest.importorskip("pyarrow", reason="检查器需要 pyarrow")
pd = pytest.importorskip("pandas", reason="写测试用特征表需要 pandas")

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from flsoh.features import (  # noqa: E402
    DataContractError,
    curve_total_variation,
    delta_q_curve,
    make_v_grid,
    segment_capacity_curve,
)

import check_features as cf  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"

#: 与 build_features.py 的默认一致：1000 点、[2.0, 3.5] V。
GRID = make_v_grid(2.0, 3.5, 1000)


def _cycle(no: int):
    """真实 MATR 循环的四个通道：V / Qd / I / Qc。"""
    return (
        np.load(FIXTURES / f"matr_c0_cyc{no}_V.npy"),
        np.load(FIXTURES / f"matr_c0_cyc{no}_Qd.npy"),
        np.load(FIXTURES / f"matr_c0_cyc{no}_I.npy"),
        np.load(FIXTURES / f"matr_c0_cyc{no}_Qc.npy"),
    )


def _delta_q(ref, cur, *, segmented: bool) -> np.ndarray:
    """循环 5 相对循环 1 的 ΔQ。``segmented=False`` 就是曾经的那个 bug。

    buggy 路径用 ``Qc``：原代码走的是 ``charge_curve()``（V + Qc），所以只有
    用 Qc 才叫**忠实复现**。换成 Qd 会得到另一个数，那就不是在测那个 bug 了。
    """
    if segmented:
        pairs = [segment_capacity_curve(v, qd, i) for v, qd, i, _qc in (ref, cur)]
    else:
        pairs = [(v, qc) for v, _qd, _i, qc in (ref, cur)]
    return delta_q_curve(*pairs[1], GRID) - delta_q_curve(*pairs[0], GRID)


def _tv_ratio(curve: np.ndarray) -> float:
    span = float(curve.max() - curve.min())
    return curve_total_variation(curve) / span if span > 0 else float("inf")


# ---------------------------------------------------------------------------
# 真实数据上的分段


class TestSegmentOnRealCycle:
    def test_cycle_contains_both_segments(self):
        """真实循环确实是两段拼的——这是那个 bug 的前提，先钉住它。"""
        v, _qd, cur, _qc = _cycle(1)
        charge = int((cur > 0).sum())
        discharge = int((cur < 0).sum())
        assert charge > 400 and discharge > 400
        assert charge + discharge < len(cur)      # 中间还有停顿点
        # 电压先升后降：整条循环不是单调的
        assert not (np.diff(v) > 0).all()
        assert not (np.diff(v) < 0).all()

    def test_discharge_segment_spans_the_grid(self):
        """修正后 [2.0, 3.5] 网格必须落在放电段值域内——否则要改网格。"""
        v, qd, cur, _qc = _cycle(1)
        sv, sq = segment_capacity_curve(v, qd, cur)
        assert sv.min() <= GRID.min() + 1e-9
        assert sv.max() >= GRID.max() - 1e-9

    def test_charge_segment_would_not_span_the_grid(self):
        """反面对照：充电段的下界在 2.02 V 以上，[2.0, 3.5] 覆盖不住。

        这条说明「改成充电段」不是免费的——网格得跟着收。选放电段的另一个
        理由就在这里：它天然覆盖既有的网格。
        """
        v, _qd, cur, qc = _cycle(1)
        sv, _sq = segment_capacity_curve(v, qc, cur, sign=+1.0)
        assert sv.min() > GRID.min()


# ---------------------------------------------------------------------------
# 那个 bug 本身


class TestTheBug:
    def test_full_cycle_is_not_smooth(self):
        """整条循环算出的「ΔQ」总变差必须大到能被检查器抓住。"""
        bad = _delta_q(_cycle(1), _cycle(5), segmented=False)
        assert _tv_ratio(bad) > 50.0, f"TV/幅度 = {_tv_ratio(bad):.1f}，本应 > 50"

    def test_segmented_is_smooth(self):
        """分段后必须光滑——这是「修正真的修好了」的判据。"""
        good = _delta_q(_cycle(1), _cycle(5), segmented=True)
        assert _tv_ratio(good) < 20.0, f"TV/幅度 = {_tv_ratio(good):.1f}，本应 < 20"

    def test_the_two_are_essentially_uncorrelated(self):
        """两条曲线必须基本不相干——这是「不是精度问题，是数据问题」的判据。"""
        bad = _delta_q(_cycle(1), _cycle(5), segmented=False)
        good = _delta_q(_cycle(1), _cycle(5), segmented=True)
        r = float(np.corrcoef(bad, good)[0, 1])
        assert abs(r) < 0.2, f"相关系数 {r:+.3f}，本应接近 0"

    def test_segmented_has_physical_amplitude(self):
        """分段后 ΔQ 的幅度必须是容量衰老的量级（Ah 的千分位），不是 1 Ah。"""
        good = _delta_q(_cycle(1), _cycle(5), segmented=True)
        span = float(good.max() - good.min())
        assert 1e-4 < span < 0.2, f"幅度 {span:.5f} Ah，不像 ΔQ"

    def test_segmented_is_not_a_flat_line(self):
        """拦「修正后变成一条平线」这种退化解：平线的 TV 也小，也能过光滑检查。

        不断言峰在哪个电压——那一测就变成复现文献数字了，而本项目只报相对差。
        这里只断言**有结构**：峰值远高于中位水平，且两端都显著低于峰。
        """
        good = _delta_q(_cycle(1), _cycle(5), segmented=True)
        peak = float(np.abs(good).max())
        assert peak > 3 * float(np.median(np.abs(good))), "曲线没有明显结构"
        assert abs(good[0]) < 0.5 * peak, "低电压端没有衰减"
        assert abs(good[-1]) < 0.5 * peak, "高电压端没有衰减"


# ---------------------------------------------------------------------------
# segment_capacity_curve 的边界


class TestSegmentEdges:
    def test_dedupes_duplicate_voltages(self):
        """重复电压只保留最后一个容量——np.interp 对重复 x 的取值不可依赖。"""
        v = np.array([3.0, 3.0, 3.0, 2.0, 2.0])
        q = np.array([0.1, 0.2, 0.3, 0.0, 0.05])
        i = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])
        sv, sq = segment_capacity_curve(v, q, i)
        assert sv.shape == (2,)
        assert sq[0] == pytest.approx(0.05)        # 2.0 V 取最后一个
        assert sq[1] == pytest.approx(0.3)         # 3.0 V 取最后一个

    def test_length_mismatch_raises(self):
        with pytest.raises(DataContractError):
            segment_capacity_curve(np.zeros(3), np.zeros(4), np.zeros(3))

    def test_too_few_points_raises(self):
        v = np.array([3.0])
        with pytest.raises(DataContractError, match="不够构造曲线"):
            segment_capacity_curve(v, np.zeros(1), np.array([-1.0]))

    def test_rest_points_are_excluded(self):
        """|I| 低于门限的点不属于任何一段。"""
        v = np.array([3.0, 3.1, 3.2, 3.3])
        q = np.array([0.0, 0.1, 0.2, 0.3])
        i = np.array([-1.0, 0.0, -1.0, -1.0])
        sv, sq = segment_capacity_curve(v, q, i)
        assert sv.shape == (3,)
        assert not np.isclose(sq, 0.1).any()

    def test_empty_after_filtering_raises(self):
        with pytest.raises(DataContractError):
            segment_capacity_curve(
                np.array([3.0, 3.1]), np.array([0.0, 0.1]), np.array([0.0, 0.0])
            )


class TestTotalVariation:
    def test_smooth_curve_has_small_tv(self):
        assert curve_total_variation(np.linspace(0.0, 1.0, 100)) == pytest.approx(1.0)

    def test_noisy_curve_has_large_tv(self):
        rng = np.random.default_rng(0)
        noisy = np.linspace(0.0, 1.0, 100) + rng.normal(0, 0.05, 100)
        # 100 点、σ=0.05 的白噪声，总变差期望约 n·σ·√(2/π) ≈ 4，
        # 所以 3 倍是有依据的下界，不是拍脑袋。
        assert curve_total_variation(noisy) > 3 * curve_total_variation(
            np.linspace(0.0, 1.0, 100)
        )

    def test_single_point_raises(self):
        with pytest.raises(DataContractError):
            curve_total_variation(np.array([1.0]))


# ---------------------------------------------------------------------------
# 检查器本身：生成器算错的话，闸门会忠实地把错的数固化下来


class TestChecker:
    @staticmethod
    def _table(tmp_path, curve: np.ndarray, name: str = "t.parquet"):
        df = pd.DataFrame({"cell_id": ["b1c0"] * 1, "cycle": [5]})
        for i, value in enumerate(curve):
            df[f"f{i}"] = [float(value)]
        path = tmp_path / name
        df.to_parquet(path)
        return path

    def test_passes_on_segmented_data(self, tmp_path):
        good = _delta_q(_cycle(1), _cycle(5), segmented=True)
        path = self._table(tmp_path, good)
        report = cf.check_feature_table(path)
        assert report["n_over_limit"] == 0
        assert report["tv_ratio_max"] < 20.0

    def test_fails_on_full_cycle_data(self, tmp_path):
        """负控：把曾经那份错数据喂进去，检查器必须报出来。"""
        bad = _delta_q(_cycle(1), _cycle(5), segmented=False)
        path = self._table(tmp_path, bad)
        report = cf.check_feature_table(path)
        assert report["n_over_limit"] == 1
        assert report["worst"][0]["cell_id"] == "b1c0"
        assert report["worst"][0]["cycle"] == 5

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(DataContractError):
            cf.check_feature_table(tmp_path / "nope.parquet")

    def test_no_feature_columns_raises(self, tmp_path):
        path = tmp_path / "empty.parquet"
        pd.DataFrame({"cell_id": ["b1c0"], "cycle": [5]}).to_parquet(path)
        with pytest.raises(DataContractError, match="一列特征都没有"):
            cf.check_feature_table(path)

    def test_constant_nonzero_row_is_flagged(self, tmp_path):
        """非零但整行常数：这才是「插值链把点配平了」的样子，必须算超限。

        零行不在此列——参考循环自身 ΔQ ≡ 0 是构造出来的，见下一条。
        """
        path = self._table(tmp_path, np.full(1000, 0.3))
        report = cf.check_feature_table(path)
        assert report["n_over_limit"] == 1
        assert report["worst"][0]["tv_ratio"] == float("inf")

    def test_reference_row_is_counted_not_flagged(self, tmp_path):
        """参考循环行（ΔQ ≡ 0）不是缺陷：每颗电池必然有一行。

        它幅度为 0，TV/幅度 没有定义。把它判成超限的话，任何
        ``--cycles`` 区间包含参考循环的合法配置都会被判红。
        """
        path = self._table(tmp_path, np.zeros(1000))
        report = cf.check_feature_table(path)
        assert report["n_reference_rows"] == 1
        assert report["n_over_limit"] == 0
        assert not report["too_many_zero_rows"]
        assert cf.main([str(path)]) == 0

    def test_all_zero_table_is_flagged(self, tmp_path):
        """整表全零必须拦住：全零行算不出 TV/幅度，不设闸门就静默通过。"""
        df = pd.DataFrame({"cell_id": ["b1c0", "b1c1"], "cycle": [1, 1]})
        for i in range(20):
            df[f"f{i}"] = [0.0, 0.0]
        path = tmp_path / "zeros.parquet"
        df.to_parquet(path)
        report = cf.check_feature_table(path)
        assert report["n_reference_rows"] == 2
        assert report["n_cells"] == 2
        # 两行零、两颗电池 → 恰好没超（每颗一个参考循环）
        assert not report["too_many_zero_rows"]
        # 再加一行零就超了：第三行不可能是任何电池的参考循环
        df3 = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        path3 = tmp_path / "zeros3.parquet"
        df3.to_parquet(path3)
        assert cf.check_feature_table(path3)["too_many_zero_rows"]
        assert cf.main([str(path3)]) == 1

    def test_cli_exit_codes(self, tmp_path):
        good = _delta_q(_cycle(1), _cycle(5), segmented=True)
        bad = _delta_q(_cycle(1), _cycle(5), segmented=False)
        assert cf.main([str(self._table(tmp_path, good, "good.parquet"))]) == 0
        assert cf.main([str(self._table(tmp_path, bad, "bad.parquet"))]) == 1
        assert cf.main([str(tmp_path / "missing.parquet")]) == 2


def _row(n: int, amp: float, ratio: float) -> np.ndarray:
    """造一行 ΔQ：线性斜坡 + 交替噪声，使 TV/幅度 ≈ ratio。

    斜坡本身的 TV = 幅度。交替噪声让每个相邻差分从 step 变成 |step ± ε|，
    ε ≫ step 时 TV ≈ (n−1)·ε，而幅度 ≈ amp + ε，解出 ε 即可指定比值。
    系数写成 ``(n−1)−ratio`` 而不是 ``2(n−1)−ratio``：相邻点是
    ``base_i`` 与 ``base_{i+1}+ε`` 交替，差分只被加了**一次** ε。
    """
    eps = ratio * amp / ((n - 1) - ratio)
    return np.linspace(0.0, amp, n) + eps * (np.arange(n) % 2)


class TestTwoTierGate:
    """判据从「逐行卡 20」改成「中位数 20 + 单行硬上限 60」。

    改的理由：TV/幅度 被 1/幅度 污染（1080 行实测 corr(TV/幅度, 1/幅度) = 0.774，
    按幅度四分组的比值中位从 8.13 单调降到 3.45）。逐行卡阈值时，**多加几颗
    低幅度电池本身**就会让闸门失败——一道会因为「数据变多」而失败的闸门，
    测的不是数据质量。下面 `test_a_tail_of_low_amplitude_rows_does_not_fail`
    把这个场景直接做成回归测试。
    """

    N = 1000

    @staticmethod
    def _table(tmp_path, rows, name="t.parquet"):
        df = pd.DataFrame({
            "cell_id": [f"c{i}" for i in range(len(rows))],
            "cycle": [5] * len(rows),
        })
        for j in range(len(rows[0])):
            df[f"f{j}"] = [float(r[j]) for r in rows]
        path = tmp_path / name
        df.to_parquet(path)
        return path

    def test_helper_produces_the_ratio_it_claims(self):
        """前提检查：造行函数真的造出了目标比值。

        少了这条，下面所有测试都可能因为「造出来的行比值是 1」而通过，
        而失败场景根本没被覆盖。
        """
        for target in (5.0, 25.0, 100.0):
            report = cf.check_feature_frame(pd.DataFrame(
                {"cell_id": ["c0"], "cycle": [5],
                 **{f"f{j}": [float(v)] for j, v in enumerate(_row(self.N, 0.01, target))}}
            ))
            assert abs(report["tv_ratio_median"] - target) / target < 0.15, (
                f"目标 {target}，实测 {report['tv_ratio_median']:.2f}"
            )

    def test_systematic_misalignment_fails_on_the_median(self, tmp_path):
        """负控：那个 bug 让所有行一起坏，中位数必须报出来。"""
        rows = [_row(self.N, 0.01, 120.0) for _ in range(40)]
        report = cf.check_feature_table(self._table(tmp_path, rows))
        assert report["median_over_limit"] is True
        assert report["tv_ratio_median"] > 100
        assert cf.report_is_ok(report) is False

    def test_a_tail_of_low_amplitude_rows_does_not_fail(self, tmp_path):
        """回归：76 行正常 + 4 行低幅度噪声行 → 必须通过。

        这 4 行在旧判据（逐行卡 20）下会让整张表失败。它们不是坏行，是
        幅度小到噪声占主导的行——真实数据里就是最小的那一档幅度。
        """
        rows = [_row(self.N, 0.03, 5.0) for _ in range(76)]
        rows += [_row(self.N, 0.008, 25.0) for _ in range(4)]
        report = cf.check_feature_table(self._table(tmp_path, rows))
        assert report["n_soft_over"] == 4
        assert report["n_over_limit"] == 0
        assert report["median_over_limit"] is False
        assert cf.report_is_ok(report) is True

    def test_one_broken_row_fails_on_the_hard_cap(self, tmp_path):
        """中位数管不到个别坏行，所以硬上限必须独立存在。"""
        rows = [_row(self.N, 0.03, 5.0) for _ in range(79)]
        rows.append(_row(self.N, 0.01, 120.0))
        report = cf.check_feature_table(self._table(tmp_path, rows))
        assert report["n_over_limit"] == 1
        assert report["median_over_limit"] is False
        assert cf.report_is_ok(report) is False
        assert report["worst"][0]["cell_id"] == "c79"

    def test_constant_row_is_still_caught(self, tmp_path):
        """整行常数 → 比值 inf，两道闸门里至少硬上限要抓住。"""
        rows = [_row(self.N, 0.03, 5.0) for _ in range(20)]
        rows.append(np.full(self.N, 0.3))
        report = cf.check_feature_table(self._table(tmp_path, rows))
        assert report["n_over_limit"] == 1
        assert cf.report_is_ok(report) is False

    def test_report_is_ok_is_the_single_source_of_truth(self):
        """三个调用点共用这一个规则，避免改判据时漏掉一处。"""
        base = {"too_many_zero_rows": False, "median_over_limit": False,
                "n_over_limit": 0}
        assert cf.report_is_ok(base) is True
        for key, value in (("too_many_zero_rows", True),
                           ("median_over_limit", True),
                           ("n_over_limit", 3)):
            assert cf.report_is_ok({**base, key: value}) is False, key


class TestLabelSignal:
    """光滑 ≠ 有信息：第三道闸门测的是「特征有没有随老化变化」。

    为什么必须单独有这道：一条常数曲线的 TV/幅度 是 0，完美光滑。所以前两道
    闸门（中位数、单行硬上限）**结构上不可能**发现「特征与标签无关」。
    2026-10-25 的错口径表恰好两者都坏，但如果只修好光滑性而取错电压区间，
    就会得到一张又光滑又无信息的表——这道闸门就是为那种情况准备的。

    真实数据上的分辨力（每颗电池取 1000 个电压点的中位 ρ，再跨电池汇总）：

    ============ =========== ================
    表            ρ 中位      ρ ≤ −0.8 的占比
    ============ =========== ================
    错口径旧表     −0.1677     0/135
    正确新表       −0.9789     61/67
    ============ =========== ================
    """

    CYCLES = list(range(5, 105, 5))
    N = 200

    @staticmethod
    def _frame(*, decay: float, jitter: float = 0.0, cells: int = 6):
        """造一张「每颗电池 K 个循环」的特征表。

        ``decay`` 是每个循环的幅度衰减比例（0.2 = 100 个循环掉 20%），
        ``jitter`` 是逐循环的独立噪声幅度。jitter 与循环号无关时，ρ ≈ 0：
        曲线仍然很光滑，但它不含任何老化信息。
        """
        rng = np.random.default_rng(0)
        base = np.linspace(0.0, 0.03, TestLabelSignal.N)
        cycles = TestLabelSignal.CYCLES
        noise = {k: rng.normal(0.0, jitter, TestLabelSignal.N) for k in cycles}
        rows = []
        for c in range(cells):
            for k in cycles:
                amp = (1.0 + 0.01 * c) * (1.0 - decay * (k - 1) / 100.0)
                values = base * amp + noise[k]
                rows.append({
                    "cell_id": f"c{c}", "cycle": k, "soh": 1.0,
                    **{f"f{j}": float(v) for j, v in enumerate(values)},
                })
        return pd.DataFrame(rows)

    def test_aging_table_passes(self):
        """正控：幅度随循环衰减 → ρ 接近 −1，通过。"""
        report = cf.check_feature_frame(self._frame(decay=0.2))
        assert report["rho_median"] < -0.9
        assert report["rho_frac_strong"] == 1.0
        assert report["label_signal_bad"] is False
        assert report["rho_undetermined"] is False
        assert cf.report_is_ok(report) is True

    def test_smooth_but_uninformative_table_fails(self):
        """负控（这道闸门存在的理由）：光滑通过、信号为零 → 必须失败。

        这张表的 TV/幅度 是 1.0（远低于 20），两道光滑性闸门都放行。
        只有 ρ 能指出「每颗电池的曲线在 20 个循环里没变过」。
        """
        report = cf.check_feature_frame(self._frame(decay=0.0, jitter=1e-6))
        assert report["median_over_limit"] is False, "前提：光滑性闸门必须放行"
        assert report["n_over_limit"] == 0, "前提：光滑性闸门必须放行"
        assert abs(report["rho_median"]) < 0.3
        assert report["label_signal_bad"] is True
        assert cf.report_is_ok(report) is False

    def test_reversed_trend_also_fails(self):
        """幅度随循环变大（比无信息更坏）同样要被拦。"""
        report = cf.check_feature_frame(self._frame(decay=-0.2))
        assert report["rho_median"] > 0.9
        assert report["label_signal_bad"] is True
        assert cf.report_is_ok(report) is False

    def test_all_constant_table_is_undetermined_not_passed(self):
        """整表常数 → 「未判定」，**不是**「通过」。

        秩相关会把「所有值相同」变成「秩 0,1,2,…」，看起来像完美正相关；
        必须显式把常数列判成 NaN。另外这里用的是 ptp 而不是 std——`np.std`
        对完全相同的值会因均值误差给出 6e-19，判不出 0（实测踩过）。
        """
        report = cf.check_feature_frame(self._frame(decay=0.0))
        assert report["rho_n_cells_judged"] == 0
        assert report["rho_undetermined"] is True
        assert report["label_signal_bad"] is False, "未判定不许写成失败"
        # report_is_ok 不把「未判定」当失败（模拟路径需要），但调用点必须
        # 自己看这个标志——build_features.py 就是这么做的。
        assert cf.report_is_ok(report) is True

    def test_report_is_ok_includes_the_label_signal(self):
        base = {"too_many_zero_rows": False, "median_over_limit": False,
                "n_over_limit": 0}
        assert cf.report_is_ok(base) is True
        assert cf.report_is_ok({**base, "label_signal_bad": True}) is False
