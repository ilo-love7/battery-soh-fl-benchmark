"""``flsoh.simulate`` 与 ``scripts/simulate_sessions.py`` 的测试。

为什么值得测
------------
模拟器是端到端联调（``run_e2e.py``）的唯一数据来源。它错的方式**不会抛异常**：

- 时间戳先降后升 → 容量积分累加重复区间，Q(V) 整体偏高而形状正常；
- 端电压覆盖不住电压网格 → 每一行都被判「覆盖不住」，表是空的但报错指向网格；
- 真值 SOH 与实验室路径不同尺度（相对出厂容量 vs 相对首循环）→ 联调出来的
  MAE 与 M1–M4 的数不能放在一起看，而两者都「看起来正常」；
- 电池之间没有差异 → 客户端异质性恒为 0，任何联邦结论都无从谈起。

四条都只在数字上留下痕迹，所以每条都要有断言。
"""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy", reason="simulate 需要 numpy")

from flsoh import segments as sg  # noqa: E402
from flsoh.features import make_v_grid  # noqa: E402
from flsoh.schema import validate_session, validate_sessions  # noqa: E402
from flsoh.simulate import (  # noqa: E402
    OCV_HI_V,
    OCV_LO_V,
    REST_POINTS,
    CellSpec,
    capacity_at,
    internal_resistance,
    open_circuit_voltage,
    simulate_battery,
    simulate_cycle,
    simulate_fleet,
    write_sessions_jsonl,
    write_truth_json,
)

GRID = make_v_grid(2.0, 3.5, 200)


def _spec(**kwargs) -> CellSpec:
    base = dict(battery_id="sim000", q0_ah=1.10, fade=0.12, r0_ohm=0.030)
    base.update(kwargs)
    return CellSpec(**base)


def _one_cycle(**kwargs):
    spec = _spec()
    rng = np.random.default_rng(0)
    return simulate_cycle(spec, 3, total_cycles=20, rng=rng, **kwargs)


# ---------------------------------------------------------------------------
# 物理模型的形状约束


class TestOpenCircuitVoltage:
    def test_monotone_increasing(self):
        soc = np.linspace(0.0, 1.0, 50)
        v = open_circuit_voltage(soc)
        assert np.all(np.diff(v) > 0), "OCV 必须随 SOC 单调上升"

    def test_endpoints(self):
        assert open_circuit_voltage(0.0) == pytest.approx(OCV_LO_V)
        assert open_circuit_voltage(1.0) == pytest.approx(OCV_HI_V)

    def test_out_of_range_soc_is_clipped(self):
        """soc_est 越界是 App 的常见故障，物理模型不能跟着外延。"""
        assert open_circuit_voltage(-0.5) == pytest.approx(OCV_LO_V)
        assert open_circuit_voltage(1.5) == pytest.approx(OCV_HI_V)

    def test_smaller_gamma_is_steeper_at_low_soc(self):
        """γ 小 → 低 SOC 端更陡。这条钉住「γ 确实起作用」，否则
        ``simulate_cycle`` 里随老化调整 γ 的那一项等于没写。"""
        soc = np.array([0.1])
        assert open_circuit_voltage(soc, gamma=0.2) > open_circuit_voltage(soc, gamma=0.5)


class TestCapacity:
    def test_first_cycle_is_below_pristine(self):
        spec = _spec()
        assert capacity_at(spec, 1, 20) < spec.q0_ah

    def test_last_cycle_hits_the_declared_fade(self):
        """``fade=0.12`` 的语义是「跑满全部循环后剩 88%」。"""
        spec = _spec(fade=0.12)
        assert capacity_at(spec, 20, 20) == pytest.approx(spec.q0_ah * 0.88)

    def test_decreasing_in_cycle_number(self):
        spec = _spec()
        values = [capacity_at(spec, k, 20) for k in range(1, 21)]
        assert np.all(np.diff(values) < 0)

    def test_fade_is_front_loaded(self):
        """平方根衰减：前一半循环掉的容量多于后一半。

        换成线性衰减时这条会失败——而那正是「模型只学平均斜率就够」的成因。
        """
        spec = _spec()
        first_half = capacity_at(spec, 1, 20) - capacity_at(spec, 10, 20)
        second_half = capacity_at(spec, 11, 20) - capacity_at(spec, 20, 20)
        assert first_half > second_half

    @pytest.mark.parametrize("bad", [dict(q0_ah=0.0), dict(fade=1.0), dict(r0_ohm=-1.0)])
    def test_invalid_spec_is_rejected(self, bad):
        with pytest.raises(ValueError):
            _spec(**bad)


class TestInternalResistance:
    def test_equals_r0_at_full_capacity(self):
        spec = _spec()
        assert internal_resistance(spec, spec.q0_ah) == pytest.approx(spec.r0_ohm)

    def test_grows_as_capacity_fades(self):
        spec = _spec()
        fresh = internal_resistance(spec, spec.q0_ah)
        aged = internal_resistance(spec, spec.q0_ah * 0.85)
        assert aged > fresh

    def test_never_negative_even_if_capacity_exceeds_pristine(self):
        """容量高于出厂值时 loss 为负，不能让内阻掉到 r0 以下。"""
        spec = _spec()
        assert internal_resistance(spec, spec.q0_ah * 1.1) == pytest.approx(spec.r0_ohm)


# ---------------------------------------------------------------------------
# 单循环：契约与可积分性


class TestSimulateCycle:
    def test_session_satisfies_schema(self):
        session, _ = _one_cycle()
        assert validate_session(session) == []

    def test_timestamps_strictly_increase(self):
        """2026-10-25 的坑：静置段写成 arange(4,0,-1)，时间先降后升。"""
        session, _ = _one_cycle()
        t = np.array([r["t"] for r in session["records"]])
        assert np.all(np.diff(t) > 0)

    def test_voltage_covers_the_feature_grid(self):
        """端电压必须盖住 [2.0, 3.5]，否则每一行特征都被判「覆盖不住」。"""
        session, _ = _one_cycle()
        v = np.array([r["V"] for r in session["records"]])
        assert v.min() <= GRID[0]
        assert v.max() >= GRID[-1]

    def test_voltage_covers_grid_even_when_heavily_aged(self):
        """重度老化时内阻涨到 0.064 Ω，1C 的压降是 0.070 V。

        OCV 上界若只留 0.06 V 余量，这里就会覆盖不住——而且是在**最后一个
        循环**才发生，前面 19 个循环全部正常。
        """
        spec = _spec(fade=0.30, r0_ohm=0.040)
        rng = np.random.default_rng(1)
        session, _ = simulate_cycle(spec, 20, total_cycles=20, rng=rng)
        v = np.array([r["V"] for r in session["records"]])
        assert v.max() >= GRID[-1]
        assert v.min() <= GRID[0]

    def test_exactly_one_discharge_segment(self):
        session, _ = _one_cycle()
        segs, report = sg.split_segments_report(session["records"])
        assert len(segs) == 1, report
        assert report["n_kept"] == 1

    def test_rest_segments_have_zero_current(self):
        session, _ = _one_cycle()
        current = np.array([r["I"] for r in session["records"]])
        assert (current[:REST_POINTS] == 0).all()
        assert (current[-REST_POINTS:] == 0).all()
        assert (current[REST_POINTS:-REST_POINTS] < 0).all(), "中段必须全程放电"

    def test_rest_voltage_is_open_circuit(self):
        """静置段电流为 0，端电压必须回到开路电压（内阻压降消失）。"""
        session, _ = _one_cycle(noise_v=0.0)
        v = np.array([r["V"] for r in session["records"]])
        assert v[0] == pytest.approx(OCV_HI_V, abs=1e-6)
        assert v[-1] == pytest.approx(OCV_LO_V, abs=1e-6)

    def test_truth_capacity_matches_the_model(self):
        session, truth = _one_cycle()
        assert truth["capacity_ah"] == pytest.approx(capacity_at(_spec(), 3, 20))
        assert truth["session_id"] == session["session_id"]
        assert truth["cycle"] == 3

    def test_same_seed_is_reproducible(self):
        a, _ = _one_cycle()
        b, _ = _one_cycle()
        assert a == b

    def test_different_seed_differs(self):
        a, _ = _one_cycle()
        spec = _spec()
        b, _ = simulate_cycle(spec, 3, total_cycles=20, rng=np.random.default_rng(99))
        assert a != b

    def test_jitter_makes_sampling_irregular(self):
        """``sampling_hint="variable"`` 必须名副其实：间隔不能是常数。"""
        session, _ = _one_cycle()
        t = np.array([r["t"] for r in session["records"]])
        dt = np.diff(t[REST_POINTS:-REST_POINTS])
        assert dt.std() > 0, "采样间隔完全没有抖动"

    def test_bad_dt_is_rejected(self):
        with pytest.raises(ValueError):
            _one_cycle(dt_s=0.0)


# ---------------------------------------------------------------------------
# 一颗电池 / 一批电池：真值的尺度


class TestSimulateBattery:
    def test_returns_one_session_per_cycle(self):
        sessions, truths = simulate_battery(_spec(), n_cycles=5)
        assert len(sessions) == 5 and len(truths) == 5

    def test_first_cycle_soh_is_exactly_one(self):
        """SOH 相对首个循环——这是与实验室路径对齐的那条定义。"""
        _, truths = simulate_battery(_spec(), n_cycles=5)
        assert truths[0]["soh"] == pytest.approx(1.0)

    def test_soh_decreases_monotonically(self):
        _, truths = simulate_battery(_spec(), n_cycles=6)
        soh = [t["soh"] for t in truths]
        assert all(b < a for a, b in zip(soh, soh[1:])), soh

    def test_last_soh_equals_capacity_ratio(self):
        _, truths = simulate_battery(_spec(fade=0.12), n_cycles=4)
        expected = capacity_at(_spec(fade=0.12), 4, 4) / capacity_at(_spec(fade=0.12), 1, 4)
        assert truths[-1]["soh"] == pytest.approx(expected)

    def test_sessions_are_valid(self):
        sessions, _ = simulate_battery(_spec(), n_cycles=3)
        report = validate_sessions(sessions)
        assert report["invalid"] == 0, report["sample_errors"][:1]


class TestSimulateFleet:
    def test_cells_differ(self):
        """电池之间必须有差异，否则客户端异质性恒为 0。"""
        _, truths = simulate_fleet(4, n_cycles=3, seed=1)
        first = [t["capacity_ah"] for t in truths if t["cycle"] == 1]
        assert len(set(round(x, 6) for x in first)) == len(first)

    def test_battery_ids_are_unique_and_ordered(self):
        sessions, _ = simulate_fleet(3, n_cycles=2, seed=2)
        ids = sorted({s["battery_id"] for s in sessions})
        assert ids == ["sim000", "sim001", "sim002"]

    def test_same_seed_same_fleet(self):
        a, ta = simulate_fleet(3, n_cycles=2, seed=5)
        b, tb = simulate_fleet(3, n_cycles=2, seed=5)
        assert a == b and ta == tb

    def test_zero_cells_is_rejected(self):
        with pytest.raises(ValueError):
            simulate_fleet(0, n_cycles=2)


# ---------------------------------------------------------------------------
# 写盘


class TestWriters:
    def test_jsonl_has_one_line_per_session(self, tmp_path):
        sessions, _ = simulate_battery(_spec(), n_cycles=3)
        path = write_sessions_jsonl(sessions, tmp_path / "s.jsonl")
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        assert json.loads(lines[0])["session_id"] == sessions[0]["session_id"]

    def test_jsonl_uses_lf_not_crlf(self, tmp_path):
        """CRLF 会在流式解析器里留下尾部 \\r，最后一条记录解析失败。"""
        sessions, _ = simulate_battery(_spec(), n_cycles=2)
        path = write_sessions_jsonl(sessions, tmp_path / "s.jsonl")
        raw = path.read_bytes()
        assert b"\r\n" not in raw
        assert raw.endswith(b"\n")

    def test_truth_roundtrips(self, tmp_path):
        _, truths = simulate_battery(_spec(), n_cycles=3)
        path = write_truth_json(truths, tmp_path / "t.json")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["n_sessions"] == 3
        assert loaded["sessions"][0]["capacity_ah"] == pytest.approx(
            truths[0]["capacity_ah"]
        )


# ---------------------------------------------------------------------------
# 与特征链的接合：这是联调真正依赖的那一环


class TestIntoFeatures:
    def test_features_track_the_truth(self):
        """ΔQ(V) 的幅度必须随 SOH 下降而单调增大。

        这条是整个模拟器存在的理由：如果特征与真值无关，``run_e2e.py`` 跑出来
        的 MAE 再小也只是模型在拟合噪声。
        """
        sessions, truths = simulate_battery(_spec(), n_cycles=6)
        feats, skipped = sg.battery_delta_q(sessions, GRID)
        assert skipped == []
        amplitude = feats.max(axis=1) - feats.min(axis=1)
        assert np.all(np.diff(amplitude) > 0), amplitude
        soh = np.array([t["soh"] for t in truths[1:]])
        # 不要求 r = −1：幅度 ∝ 容量差（√ 衰减），而 SOH 是线性比，两者不是
        # 同一个函数形式。断言「强负相关」而不是「某个具体的 r」，否则这条测试
        # 就变成在复现模拟器自己的公式了。
        assert np.corrcoef(amplitude, soh)[0, 1] < -0.95, "幅度与 SOH 应强负相关"

    def test_no_session_is_dropped(self):
        """一份都不能丢：丢一份就意味着那一段的端电压没盖住网格。"""
        sessions, _ = simulate_fleet(3, n_cycles=4, seed=3)
        for battery in ("sim000", "sim001", "sim002"):
            own = [s for s in sessions if s["battery_id"] == battery]
            feats, skipped = sg.battery_delta_q(own, GRID)
            assert skipped == [], f"{battery}: {skipped}"
            assert feats.shape[0] == len(own) - 1


# ---------------------------------------------------------------------------
# 命令行


class TestCli:
    def test_writes_both_files(self, tmp_path):
        import simulate_sessions

        out = tmp_path / "s.jsonl"
        truth = tmp_path / "t.json"
        code = simulate_sessions.main(
            ["--cells", "2", "--cycles", "3", "--out", str(out), "--truth", str(truth)]
        )
        assert code == 0
        assert out.is_file() and truth.is_file()
        assert len(out.read_text(encoding="utf-8").splitlines()) == 6

    def test_invalid_sessions_fail_the_run(self, tmp_path, monkeypatch, capsys):
        """负控：模拟器产出违反契约的数据时，CLI 必须非零退出。

        没有这道闸门的话，坏数据会被写进 JSONL，而 ``run_e2e.py`` 不会替它
        再查一遍——联调会一路跑到底，产出一张看起来正常的表。
        """
        import simulate_sessions

        bad_session, bad_truth = _one_cycle()
        bad_session["records"][1]["t"] = bad_session["records"][0]["t"]  # 时间不递增
        bad_truth["soh"] = 1.0

        monkeypatch.setattr(
            simulate_sessions, "simulate_fleet",
            lambda *a, **k: ([bad_session], [bad_truth]),
        )
        out = tmp_path / "s.jsonl"
        code = simulate_sessions.main(
            ["--cells", "1", "--cycles", "1", "--out", str(out), "--truth", str(tmp_path / "t.json")]
        )
        assert code == 1
        assert not out.is_file(), "校验失败时不能留下半份数据"
        assert "Schema v1" in capsys.readouterr().err

    def test_no_validate_skips_the_gate(self, tmp_path, monkeypatch):
        """``--no-validate`` 存在是为了在契约演进时能先跑通，但它必须显式给。"""
        import simulate_sessions

        bad_session, bad_truth = _one_cycle()
        bad_session["records"][1]["t"] = bad_session["records"][0]["t"]
        bad_truth["soh"] = 1.0
        monkeypatch.setattr(
            simulate_sessions, "simulate_fleet",
            lambda *a, **k: ([bad_session], [bad_truth]),
        )
        out = tmp_path / "s.jsonl"
        code = simulate_sessions.main(
            ["--cells", "1", "--cycles", "1", "--out", str(out),
             "--truth", str(tmp_path / "t.json"), "--no-validate"]
        )
        assert code == 0
        assert out.is_file()
