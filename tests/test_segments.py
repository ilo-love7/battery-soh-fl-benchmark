"""``segments.py`` 的测试：App 路径的特征提取。

为什么值得单独测
----------------
这个模块把「手机采样」变成「和实验室同一条管线的输入」。它有三种**不会抛
异常**的失效方式，而且都只体现在「数字有点怪」上：

1. **段切错**（静置点算进放电段）→ 容量积分掺进静置时间，Q(V) 整体偏移；
2. **积分用左端点而非梯形** → 可变采样率下稀疏段被高估；
3. **容量没按段重置** → 第二段之后全部系统性偏高，而形状完全正常。

所以测试不是「能跑通」，而是逐条钉住这三种失效不会发生，并且用
``app_roundtrip_error`` 量化 App 路径比实验室路径多付多少。
"""

from __future__ import annotations

import pytest

# importorskip 必须是本文件的第一个 import（连同 stdlib import 也不能有第三方
# 的写在它上面）：默认 CI job 只装 pytest，`import numpy` 写在上面会让
# collection 阶段 ModuleNotFoundError，报出来是「3 errors」而不是「全部
# skipped」——这个坑在 test_matr.py 踩过一次。
np = pytest.importorskip("numpy", reason="segments 需要 numpy")

from flsoh.features import (  # noqa: E402
    DataContractError,
    delta_q_curve,
    delta_q_features,
    make_v_grid,
)
from flsoh import segments as sg  # noqa: E402

GRID = make_v_grid(2.0, 3.5, 200)


def _rest_records(*, n=10, dt=30.0, t_start=1_700_000_000.0):
    """合成一段静置：电流低于 ``REST_THRESHOLD_A``，必须把放电段断开。"""
    t = t_start + np.arange(n) * dt
    return [
        {"t": float(t[k]), "V": 2.0, "I": 0.0, "T": 25.0, "soc_est": 0.0}
        for k in range(n)
    ]


def _discharge_session(*, n=120, dt=30.0, current=-1.2, t_start=1_700_000_000.0,
                       v0=3.5, v1=2.0, with_rest=False):
    """合成一次完整放电的采样记录。"""
    q = np.linspace(0.0, 1.0, n)              # Ah
    v = np.linspace(v0, v1, n)                # 简化为线性压降
    t = t_start + np.arange(n) * dt
    recs = []
    for k in range(n):
        i = current
        if with_rest and 40 <= k < 60:
            i = 0.0                            # 中间静置 20 个点
        recs.append({"t": float(t[k]), "V": float(v[k]), "I": float(i),
                     "T": 25.0, "soc_est": float(1.0 - k / (n - 1))})
    return recs


class TestSplitSegments:
    def test_one_clean_discharge_is_one_segment(self):
        segs = sg.split_segments(_discharge_session())
        assert len(segs) == 1
        assert segs[0].n_points == 120
        assert segs[0].mean_current == pytest.approx(-1.2)

    def test_charge_and_discharge_are_separate(self):
        recs = _discharge_session(n=60) + _discharge_session(
            n=60, current=+1.2, t_start=1_700_000_000.0 + 60 * 30.0, v0=2.0, v1=3.5
        )
        segs = sg.split_segments(recs)
        assert len(segs) == 1                    # 只认放电
        assert segs[0].mean_current < 0

    def test_rest_splits_the_segment(self):
        """静置必须断开段：把静置算进放电会让容量积分掺进静置时间。"""
        recs = _discharge_session(with_rest=True)
        segs = sg.split_segments(recs)
        assert len(segs) == 2
        assert segs[0].stop == 40 and segs[1].start == 60

    def test_short_segments_are_dropped_and_counted(self):
        recs = _discharge_session(n=5)          # 少于 min_points=8
        segs, report = sg.split_segments_report(recs)
        assert segs == []
        assert report["n_candidates"] == 1
        assert report["dropped"][0]["points"] == 5

    def test_report_counts_everything(self):
        """两段放电夹一段静置：候选 2 段、保留 2 段，记录一条不少。

        静置是**必须的**：两段放电首尾相接、电流全程同号时，段切分会把它们
        并成一段（那是对的——中间没有断开过），于是这条测试什么也没测。
        """
        recs = (_discharge_session(n=30)
                + _rest_records(n=10, t_start=1_700_000_000.0 + 30 * 30.0)
                + _discharge_session(n=100, t_start=1_700_000_000.0 + 40 * 30.0))
        segs, report = sg.split_segments_report(recs)
        assert len(segs) == 2
        assert report["n_records"] == 140
        assert report["n_kept"] == 2
        assert report["n_candidates"] == 2
        assert report["dropped"] == []
        # 两段各自的边界：第二段必须从静置之后开始
        assert segs[0].stop == 30 and segs[1].start == 40

    def test_empty_records_raise(self):
        with pytest.raises(DataContractError):
            sg.split_segments([])


class TestIntegrateCapacity:
    def test_constant_current_gives_expected_capacity(self):
        t = np.arange(0, 3601, 60.0)             # 1 小时
        cur = np.full(t.shape, -1.0)
        q = sg.integrate_capacity(t, cur)
        assert q[0] == 0.0
        assert q[-1] == pytest.approx(1.0)        # 1 A × 1 h = 1 Ah
        assert np.all(np.diff(q) > 0)

    def test_trapezoid_not_left_endpoint(self):
        """可变采样率下左端点会高估稀疏段——梯形与左端点必须不同。"""
        t = np.array([0.0, 1.0, 100.0, 101.0])
        cur = np.array([-1.0, -1.0, -1.0, -1.0])
        trap = sg.integrate_capacity(t, cur)
        left = np.concatenate([[0.0], np.cumsum(cur[:-1] * np.diff(t))]) / 3600.0
        assert not np.allclose(trap, left)

    def test_positive_current_contributes_nothing(self):
        """充电期间容量不该倒退（段内防御）。

        梯形积分在「−1 A → +5 A」这一步只计半步，所以 q 仍然上升——这里
        要守的是**单调不减**，不是「完全不动」。
        """
        t = np.arange(0, 4, 1.0)
        cur = np.array([-1.0, +5.0, -1.0, -1.0])
        q = sg.integrate_capacity(t, cur)
        assert np.all(np.diff(q) >= 0)
        # 正电流那一步的贡献必须小于两步全是 −1 A 的情况
        full = sg.integrate_capacity(t, np.full(4, -1.0))
        assert q[-1] < full[-1]

    def test_non_increasing_time_raises(self):
        with pytest.raises(DataContractError):
            sg.integrate_capacity(np.array([0.0, 0.0]), np.array([-1.0, -1.0]))

    def test_length_mismatch_raises(self):
        with pytest.raises(DataContractError):
            sg.integrate_capacity(np.array([0.0, 1.0]), np.array([-1.0]))


class TestSegmentCurve:
    def test_recovers_the_input_curve(self):
        """无噪声、密集采样下，App 路径必须还原输入的 Q(V)。"""
        v = np.linspace(3.5, 2.0, 200)
        q = np.linspace(0.0, 1.0, 200)
        recs = sg.degrade_to_records(v, q, sample_dt_s=1e-9)
        segs = sg.split_segments(recs, min_points=2, min_duration_s=0.0)
        recovered = sg.segment_curve(recs, segs[0], GRID)
        clean = delta_q_curve(v, q, GRID)
        assert np.max(np.abs(recovered - clean)) < 1e-6

    def test_uses_the_shared_delta_q_curve(self):
        """plan §8.2 的硬约束：App 与仿真共用同一份代码。

        这条测试守的是边界本身——如果哪天有人在本模块里重写了一遍插值，
        这里会红。
        """
        assert sg.segment_curve.__module__ == "flsoh.segments"
        # 同一个 v_grid、同一组 (V, Q)，两条路径必须一致。用 allclose 而不是
        # array_equal：记录里的时间戳是「容量反推 dQ/dt 再积分」得来的，
        # 浮点往返有 1e-16 量级差，那不是两条路径不同。
        v = np.linspace(3.5, 2.0, 50)
        q = np.linspace(0.0, 1.0, 50)
        recs = sg.degrade_to_records(v, q, sample_dt_s=1e-9)
        segs = sg.split_segments(recs, min_points=2, min_duration_s=0.0)
        got = sg.segment_curve(recs, segs[0], GRID)
        assert np.allclose(got, delta_q_curve(v, q, GRID), rtol=0, atol=1e-12)
        assert np.max(np.abs(got - delta_q_curve(v, q, GRID))) < 1e-12

    def test_grid_outside_range_raises(self):
        """电压窗口覆盖不住必须报错，不外延。"""
        v = np.linspace(3.0, 2.5, 50)
        q = np.linspace(0.0, 1.0, 50)
        recs = sg.degrade_to_records(v, q, sample_dt_s=1e-9)
        segs = sg.split_segments(recs, min_points=2, min_duration_s=0.0)
        with pytest.raises(DataContractError):
            sg.segment_curve(recs, segs[0], GRID)


class TestBatteryDeltaQ:
    def _sessions(self, n_cycles=4):
        out = []
        for c in range(n_cycles):
            v = np.linspace(3.5, 2.0, 120)
            q = np.linspace(0.0, 1.0 - 0.05 * c, 120)
            out.append({
                "session_id": f"s{c}",
                "start_ts": 1_700_000_000 + c * 7200,
                "end_ts": 1_700_000_000 + c * 7200 + 3600,
                "records": sg.degrade_to_records(v, q, sample_dt_s=1e-9,
                                                 t_start=1_700_000_000 + c * 7200),
            })
        return out

    def test_shape_and_reference_cycle(self):
        feats, skipped = sg.battery_delta_q(self._sessions(4), GRID)
        assert feats.shape == (3, len(GRID))       # 参考循环自身不产生特征
        assert skipped == []

    def test_matches_a_direct_delta_q_features_call(self):
        """硬约束：App 路径必须和实验室路径产出同一个函数的结果。

        这里手工构造同一批 (V, Q) 对，一边走 ``battery_delta_q``，一边直接
        调 ``delta_q_features``，逐位比对。若哪天有人在本模块里重写了一遍
        差分，这条会红。
        """
        sessions = self._sessions(4)
        got, _ = sg.battery_delta_q(sessions, GRID)
        pairs = []
        for s in sessions:
            segs = sg.split_segments(s["records"])
            longest = max(segs, key=lambda x: x.duration_s)
            pairs.append(sg.segment_pair(s["records"], longest))
        want = delta_q_features(pairs, GRID, reference_index=0)
        assert np.allclose(got, want, rtol=0, atol=1e-12)
        # 参考循环被排除，且第二行起对应第 2、3、4 段会话
        assert got.shape == (3, len(GRID))

    def test_skipped_sessions_are_reported(self):
        """覆盖不住窗口的会话必须逐条报出来，不静默跳过。

        4 个会话里坏掉 1 个 → 3 条曲线 → 参考循环自身不产生特征 → 2 行。
        这条同时钉住「坏一个不等于整颗电池报废」：``delta_q_features`` 是对
        整个列表一次性插值的，抛异常时不说是哪一段，所以跳过必须在进列表
        之前就做完。
        """
        sessions = self._sessions(4)
        sessions[1]["records"] = sg.degrade_to_records(
            np.linspace(3.0, 2.5, 50), np.linspace(0.0, 1.0, 50),
            sample_dt_s=1e-9, t_start=1_700_000_000 + 7200,
        )
        feats, skipped = sg.battery_delta_q(sessions, GRID)
        assert feats.shape[0] == 2
        assert len(skipped) == 1 and "s1" in skipped[0]
        assert "覆盖不住" in skipped[0]

    def test_all_sessions_skipped_raises(self):
        sessions = [{
            "session_id": "s0",
            "records": sg.degrade_to_records(
                np.linspace(3.0, 2.5, 20), np.linspace(0.0, 1.0, 20),
                sample_dt_s=1e-9),
        }]
        with pytest.raises(DataContractError):
            sg.battery_delta_q(sessions, GRID)

    def test_reference_index_out_of_range(self):
        with pytest.raises(DataContractError):
            sg.battery_delta_q(self._sessions(2), GRID, reference_index=9)

    def test_several_bad_sessions_are_all_reported(self):
        """坏多个也必须逐条报，且不影响剩下的会话。"""
        sessions = self._sessions(5)
        for c in (1, 3):
            sessions[c]["records"] = sg.degrade_to_records(
                np.linspace(3.0, 2.5, 50), np.linspace(0.0, 1.0, 50),
                sample_dt_s=1e-9, t_start=1_700_000_000 + c * 7200,
            )
        feats, skipped = sg.battery_delta_q(sessions, GRID)
        assert feats.shape[0] == 2          # 5 - 2 坏 - 1 参考 = 2
        assert [s.split(":")[0] for s in skipped] == ["s1", "s3"]

    def test_empty_session_is_reported_not_fatal(self):
        """空 records 的会话记一条跳过，不让整颗电池崩。"""
        sessions = self._sessions(3)
        sessions[1]["records"] = []
        feats, skipped = sg.battery_delta_q(sessions, GRID)
        assert feats.shape[0] == 1
        assert len(skipped) == 1 and "s1" in skipped[0]

    def test_grid_edge_is_inclusive(self):
        """曲线恰好等于网格边界时必须通过（容差 1e-9，不是严格小于）。"""
        sessions = self._sessions(2)
        for s in sessions:
            segs = sg.split_segments(s["records"])
            longest = max(segs, key=lambda x: x.duration_s)
            v, _q = sg.segment_pair(s["records"], longest)
            # 数据恰好覆盖 [2.0, 3.5] 时不报「覆盖不住」
            assert sg._spans_grid(v, GRID)
            # 网格往外扩一点就必须报
            assert not sg._spans_grid(v, np.linspace(1.99, 3.5, 10))


class TestRoundtrip:
    def test_dense_sampling_is_nearly_lossless(self):
        v = np.linspace(3.5, 2.0, 400)
        q = np.linspace(0.0, 1.0, 400)
        rep = sg.app_roundtrip_error(v, q, GRID, sample_dt_s=1e-9)
        assert rep["relative"] < 1e-6

    def test_coarse_sampling_costs_something(self):
        """采样变稀必须付出代价——否则这条链没有信息量。"""
        v = np.linspace(3.5, 2.0, 2000)
        q = np.linspace(0.0, 1.0, 2000)
        dense = sg.app_roundtrip_error(v, q, GRID, sample_dt_s=1e-9)
        coarse = sg.app_roundtrip_error(v, q, GRID, sample_dt_s=600.0)
        assert coarse["relative"] > dense["relative"]
        assert coarse["n_records"] < dense["n_records"]

    def test_noise_costs_more_than_sampling_alone(self):
        v = np.linspace(3.5, 2.0, 2000)
        q = np.linspace(0.0, 1.0, 2000)
        clean = sg.app_roundtrip_error(v, q, GRID, sample_dt_s=60.0)
        noisy = sg.app_roundtrip_error(v, q, GRID, sample_dt_s=60.0,
                                       voltage_noise_v=0.005)
        assert noisy["relative"] > clean["relative"]

    def test_extremely_coarse_sampling_is_reported_not_silent(self):
        """稀到曲线覆盖不住网格时，返回 inf 和一个诚实的 0。

        ``degrade_to_records`` 在这种采样率下会兜底补到 2 个点，所以
        ``split_segments`` **确实切出了一段**——但那段覆盖不住 [2.0, 3.5]。
        这里钉住的是：不许把「切出来了」当成「恢复出了一个循环」，
        ``n_segments`` 必须与 ``relative == inf`` 一致地报 0。
        """
        v = np.linspace(3.5, 2.0, 5000)
        q = np.linspace(0.0, 1.0, 5000)
        rep = sg.app_roundtrip_error(v, q, GRID, sample_dt_s=1e9)
        assert rep["relative"] == float("inf")
        assert rep["rmse"] == float("inf")
        assert rep["n_segments"] == 0
