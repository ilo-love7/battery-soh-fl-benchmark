"""`scripts/rerun_all.py` 的守门测试。

这个脚本的价值全在"反推不出来就报错"上。它一旦静默降级——漏一个
`--server-tau`、把 `--mu 0.1` 换成默认值、把不存在的划分文件当成"跳过"——
重跑出来的 172 个数字就全是别的实验，而文件名、日志、退出码都正常。所以
这里的测试不测"能不能跑通"，测的是**每种坏输入都会被拒绝**。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import rerun_all as ra


def _fed(**over):
    """一个最小可用的联邦 run。"""
    run = {
        "run_id": "fedavg__iid__le10__seed0",
        "strategy": "fedavg",
        "mu": 0.0,
        "n_clients": 23,
        "rounds": 100,
        "local_epochs": 10,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "seed": 0,
        "partition_json": "results/partition-mvp-k5-iid.json",
        "final_mae": 0.00131,
        "wall_clock_s": 300.0,
    }
    run.update(over)
    return run


def _baseline(strategy, seed, **over):
    run = {
        "run_id": f"{'c0__centralized' if strategy == 'centralized' else 'c1__local'}__seed{seed}",
        "strategy": strategy,
        "n_clients": None if strategy == "centralized" else 23,
        "seed": seed,
        "epochs": 300,
        "learning_rate": 0.001,
        "final_mae": 0.00156,
        "wall_clock_s": 73.0,
        "notes": "test_standardizer=test; optimizer=adam; label_center=SOH_CENTER",
    }
    run.update(over)
    return run


def _pair(argv: list[str], flag: str) -> str:
    """取 `--flag value` 里的 value，没有就抛错（比返回 None 更早暴露）。"""
    return argv[argv.index(flag) + 1]


# ---------------------------------------------------------------- 数字格式


class TestNumberFormatting:
    def test_small_float_round_trips_exactly(self):
        # `:g` 会把 1e-06 写成 1e-06（还行）但把 1.23456789e-06 写成 1.23457e-06。
        # τ 扫的正是 1e-6 与 1e-5 两档，差一位就是另一档。
        for value in (1e-6, 1e-5, 0.001, 0.01, 0.1, 1.23456789e-6):
            assert float(ra._num(value)) == value

    def test_zero_stays_zero(self):
        # `--noise-multiplier 0` 是"只裁剪不加噪"的对照组，不能变成"不开 DP"。
        assert float(ra._num(0.0)) == 0.0

    def test_long_decimal_is_not_truncated(self):
        assert float(ra._num(0.0010000001)) == 0.0010000001


# ---------------------------------------------------------------- 联邦命令


class TestFederatedCommand:
    def test_tag_is_the_run_id_verbatim(self):
        # 文件名必须逐字对上：plan §10.2 的表与图 1–3 都按 run_id 分组，
        # 差一个字符就是"少一行"而不是"报错"。
        argv = ra.build_federated_command(
            _fed(run_id="fedavg__iid_k10__seed3"), config="c.toml",
            features="f.parquet", out="results",
        )
        assert _pair(argv, "--tag") == "fedavg__iid_k10__seed3"

    def test_carries_every_training_hyperparameter(self):
        argv = ra.build_federated_command(
            _fed(), config="c.toml", features="f.parquet", out="results"
        )
        assert _pair(argv, "--rounds") == "100"
        assert _pair(argv, "--local-epochs") == "10"
        assert _pair(argv, "--lr") == "0.001"
        assert _pair(argv, "--optimizer") == "adam"
        assert _pair(argv, "--seed") == "0"
        assert _pair(argv, "--partition-json") == "results/partition-mvp-k5-iid.json"

    def test_fedavg_does_not_get_a_mu_flag(self):
        # FedAvg 传 --mu 0 会走到 proximal 分支之外，但更重要的是别传错值。
        argv = ra.build_federated_command(
            _fed(), config="c.toml", features="f.parquet", out="results"
        )
        assert "--mu" not in argv

    def test_fedprox_always_passes_mu(self):
        # 不传的话脚本退回配置里 proximal_mu 的最后一档，而 M2 扫过
        # 0.01 与 0.1 两档——退回哪一档都会把那一组换成另一组。
        argv = ra.build_federated_command(
            _fed(strategy="fedprox", mu=0.01), config="c.toml",
            features="f.parquet", out="results",
        )
        assert _pair(argv, "--mu") == "0.01"

    def test_fedadam_carries_all_four_server_hyperparameters(self):
        # 默认值随 flwr 版本漂移；M3 ② 扫过 τ=1e-6 / 1e-5，缺 τ 就等于把
        # 那一档换成 1e-3。
        argv = ra.build_federated_command(
            _fed(strategy="fedadam", server_lr=0.01, server_beta_1=0.9,
                 server_beta_2=0.99, server_tau=1e-5),
            config="c.toml", features="f.parquet", out="results",
        )
        assert _pair(argv, "--server-lr") == "0.01"
        assert _pair(argv, "--server-beta1") == "0.9"
        assert _pair(argv, "--server-beta2") == "0.99"
        assert float(_pair(argv, "--server-tau")) == 1e-5

    def test_dp_triplet_is_passed_together(self):
        argv = ra.build_federated_command(
            _fed(noise_multiplier=0.0, max_grad_norm=1.0, target_delta=1e-5),
            config="c.toml", features="f.parquet", out="results",
        )
        assert float(_pair(argv, "--noise-multiplier")) == 0.0
        assert float(_pair(argv, "--max-grad-norm")) == 1.0
        assert float(_pair(argv, "--target-delta")) == 1e-5

    def test_non_dp_run_has_no_dp_flags(self):
        argv = ra.build_federated_command(
            _fed(), config="c.toml", features="f.parquet", out="results"
        )
        assert "--noise-multiplier" not in argv


# ---------------------------------------------------------------- 基线命令


class TestBaselineCommand:
    def test_sixteen_baseline_runs_become_one_command(self):
        runs = [_baseline("centralized", s) for s in range(8)]
        runs += [_baseline("local", s) for s in range(8)]
        argv = ra.build_baseline_command(
            runs, config="c.toml", features="f.parquet",
            partition="p.json", out="results",
        )
        assert _pair(argv, "--seeds") == "0,1,2,3,4,5,6,7"
        assert _pair(argv, "--epochs") == "300"

    def test_optimizer_is_read_from_notes(self):
        # C0/C1 的 optimizer 没平铺到顶层，只在 notes 里。读不出来就不该猜 adam。
        assert ra._baseline_optimizer(_baseline("centralized", 0)) == "adam"
        assert ra._baseline_optimizer({"notes": "别的东西"}) == ""

    def test_mixed_lr_is_refused(self):
        runs = [_baseline("centralized", 0), _baseline("centralized", 1, learning_rate=0.01)]
        with pytest.raises(ValueError, match="不唯一"):
            ra.baseline_params(runs)

    def test_unreadable_optimizer_is_refused(self):
        runs = [_baseline("centralized", 0, notes="没有优化器信息")]
        with pytest.raises(ValueError, match="不唯一"):
            ra.baseline_params(runs)


# ---------------------------------------------------------------- 预检


class TestPreflight:
    def _write_partition(self, path: Path, n_clients: int = 23) -> Path:
        payload = {
            "partition": {f"client-{i}": [f"cell-{i}"] for i in range(n_clients)},
            "test_cells": ["t0"],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_missing_field_is_reported_not_skipped(self, tmp_path):
        features = tmp_path / "f.parquet"
        features.write_bytes(b"x")
        part = self._write_partition(tmp_path / "p.json")
        run = _fed(partition_json=str(part))
        del run["rounds"]
        jobs, problems = ra.preflight(
            {"r": run}, features=features, baseline_partition=part
        )
        assert not jobs
        assert any("缺字段" in p and "rounds" in p for p in problems)

    def test_missing_partition_file_is_reported(self, tmp_path):
        features = tmp_path / "f.parquet"
        features.write_bytes(b"x")
        other = self._write_partition(tmp_path / "p.json")
        jobs, problems = ra.preflight(
            {"r": _fed(partition_json=str(tmp_path / "nope.json"))},
            features=features, baseline_partition=other,
        )
        assert not jobs
        assert any("划分文件不存在" in p for p in problems)

    def test_fedadam_without_tau_is_refused(self, tmp_path):
        features = tmp_path / "f.parquet"
        features.write_bytes(b"x")
        part = self._write_partition(tmp_path / "p.json")
        run = _fed(strategy="fedadam", partition_json=str(part),
                   server_lr=0.01, server_beta_1=0.9, server_beta_2=0.99)
        jobs, problems = ra.preflight(
            {"r": run}, features=features, baseline_partition=part
        )
        assert not jobs
        assert any("server_tau" in p for p in problems)

    def test_dp_run_without_target_delta_is_refused(self, tmp_path):
        features = tmp_path / "f.parquet"
        features.write_bytes(b"x")
        part = self._write_partition(tmp_path / "p.json")
        run = _fed(partition_json=str(part), noise_multiplier=0.5, max_grad_norm=1.0)
        jobs, problems = ra.preflight(
            {"r": run}, features=features, baseline_partition=part
        )
        assert not jobs
        assert any("target_delta" in p for p in problems)

    def test_unknown_strategy_is_refused(self, tmp_path):
        features = tmp_path / "f.parquet"
        features.write_bytes(b"x")
        part = self._write_partition(tmp_path / "p.json")
        jobs, problems = ra.preflight(
            {"r": _fed(strategy="fedbn", partition_json=str(part))},
            features=features, baseline_partition=part,
        )
        assert not jobs
        assert any("未知策略" in p for p in problems)

    def test_c1_client_count_must_match_the_partition(self, tmp_path):
        # C1 是"每客户端各训各的、指标取均值"。划分文件选错时客户端数会变，
        # 而结果照样写出来——只是换了个实验。
        features = tmp_path / "f.parquet"
        features.write_bytes(b"x")
        part = self._write_partition(tmp_path / "p.json", n_clients=12)
        runs = {r["run_id"]: r for r in
                [_baseline("centralized", 0), _baseline("local", 0)]}
        jobs, problems = ra.preflight(
            runs, features=features, baseline_partition=part
        )
        assert not jobs
        assert any("n_clients=23" in p and "12" in p for p in problems)

    def test_healthy_inputs_produce_one_job_per_federated_run_plus_one(self, tmp_path):
        features = tmp_path / "f.parquet"
        features.write_bytes(b"x")
        part = self._write_partition(tmp_path / "p.json")
        runs = {
            "a": _fed(run_id="a", partition_json=str(part)),
            "b": _fed(run_id="b", partition_json=str(part), seed=1),
            "c0": _baseline("centralized", 0),
            "c1": _baseline("local", 0),
        }
        jobs, problems = ra.preflight(
            runs, features=features, baseline_partition=part
        )
        assert problems == []
        assert len(jobs) == 3
        assert [j["kind"] for j in jobs] == ["federated", "federated", "baseline"]
        assert jobs[2]["run_ids"] == ["c0__centralized__seed0", "c1__local__seed0"]


# ---------------------------------------------------------------- 特征表闸门


class TestFeatureFreshness:
    def _setup(self, tmp_path, part_mtime: float, table_mtime: float):
        table = tmp_path / "features.parquet"
        table.write_bytes(b"x")
        parts = tmp_path / "features.parquet.parts"
        parts.mkdir()
        part = parts / "b1c0.parquet"
        part.write_bytes(b"x")
        os.utime(part, (part_mtime, part_mtime))
        os.utime(table, (table_mtime, table_mtime))
        return table

    def test_newer_parts_are_refused(self, tmp_path):
        # 分片比汇总表新 = 重建还没汇总。这时开跑等于又跑一遍旧表，而命令、
        # 日志、退出码全都正常——本轮最危险的失败模式。
        table = self._setup(tmp_path, part_mtime=2000, table_mtime=1000)
        problems = ra.check_features_freshness(table, allow_stale=False)
        assert problems and "重建还没汇总" in problems[0]

    def test_newer_parts_can_be_explicitly_allowed(self, tmp_path, capsys):
        table = self._setup(tmp_path, part_mtime=2000, table_mtime=1000)
        assert ra.check_features_freshness(table, allow_stale=True) == []
        assert "WARN" in capsys.readouterr().out

    def test_fresh_table_passes(self, tmp_path):
        table = self._setup(tmp_path, part_mtime=1000, table_mtime=2000)
        assert ra.check_features_freshness(table, allow_stale=False) == []

    def test_no_parts_dir_passes(self, tmp_path):
        table = tmp_path / "features.parquet"
        table.write_bytes(b"x")
        assert ra.check_features_freshness(table, allow_stale=False) == []


class TestFeatureVersion:
    def test_mismatch_is_refused(self, tmp_path):
        table = tmp_path / "features.parquet"
        table.write_bytes(b"x")
        (tmp_path / "features.spec.json").write_text(
            json.dumps({"feature_version": "fullcycle-v1"}), encoding="utf-8"
        )
        problems = ra.check_feature_version(table, "discharge-v2")
        assert problems and "fullcycle-v1" in problems[0]

    def test_match_passes(self, tmp_path):
        table = tmp_path / "features.parquet"
        table.write_bytes(b"x")
        (tmp_path / "features.spec.json").write_text(
            json.dumps({"feature_version": "discharge-v2"}), encoding="utf-8"
        )
        assert ra.check_feature_version(table, "discharge-v2") == []

    def test_missing_spec_is_only_fatal_when_expected(self, tmp_path):
        table = tmp_path / "features.parquet"
        table.write_bytes(b"x")
        assert ra.check_feature_version(table, None) == []
        assert ra.check_feature_version(table, "discharge-v2")


# ---------------------------------------------------------------- 解释器与线程


class TestTorchVersionGate:
    """换 torch 版本会改数值，而且从数字上完全看不出来。

    实测 2026-10-25：同一份特征表、同一 seed，torch 2.10.0 得 C0 MAE
    0.0015666428208351135，torch 2.14.1 得 0.0015434211492538453。用 2.10.0
    重跑能逐位复现记录值，2.14.1 不能——所以这不是噪声。
    """

    def test_mismatch_is_refused(self, monkeypatch):
        monkeypatch.setattr(ra, "probe_torch_version", lambda python: "2.14.1+cpu")
        problems = ra.check_torch_version({"a": {"torch_version": "2.10.0+cpu"}}, "py")
        assert problems and "2.14.1+cpu" in problems[0]

    def test_match_passes(self, monkeypatch):
        monkeypatch.setattr(ra, "probe_torch_version", lambda python: "2.10.0+cpu")
        assert ra.check_torch_version({"a": {"torch_version": "2.10.0+cpu"}}, "py") == []

    def test_runs_without_a_recorded_version_are_tolerated(self, monkeypatch):
        monkeypatch.setattr(ra, "probe_torch_version", lambda python: "2.10.0+cpu")
        assert ra.check_torch_version({"a": {}}, "py") == []

    def test_unimportable_torch_is_refused(self, monkeypatch):
        # 导不进 torch 时不能"默认没问题"——那样会拿一个装不上 torch 的
        # 解释器去跑，然后每个 run 都失败。
        monkeypatch.setattr(ra, "probe_torch_version", lambda python: None)
        problems = ra.check_torch_version({"a": {"torch_version": "2.10.0+cpu"}}, "py")
        assert problems and "import torch 失败" in problems[0]

    def test_real_probe_returns_none_for_a_bogus_interpreter(self):
        assert ra.probe_torch_version("definitely-not-a-python-xyz") is None


class TestThreadCountGate:
    """线程数改浮点累加顺序，实测 C0 在 24 与 1 线程下差 2.1%。

    早期结果没有这个字段，那时不检查——也不假装检查过。
    """

    def test_mismatch_is_refused(self):
        runs = {"a": {"torch_num_threads": 24}}
        problems = ra.check_thread_count(runs, 1)
        assert problems and "24" in problems[0]

    def test_match_passes(self):
        assert ra.check_thread_count({"a": {"torch_num_threads": 1}}, 1) == []

    def test_absent_field_is_not_a_failure(self):
        assert ra.check_thread_count({"a": {"torch_version": "2.10.0+cpu"}}, 1) == []


class TestProbedThreadCount:
    """「记录一致」不等于「真的钉住了」：torch 会把 OMP_NUM_THREADS 钳掉。

    实测（本机 24 核、torch 2.10.0+cpu）：请求 24，`torch.get_num_threads()`
    返回 **16**。所以 `--threads 24` 是个钉不住的值——`pinned_env` 设了 24，
    torch 只给 16，产物如实记 16，然后全部跑完之后 `check_thread_count` 才报
    不一致（如果它报的话）。

    代价不是时间而是数值：同一个 run、同一份表、同一个 torch，
    1 线程 → 0.0021398472785949707，16 线程 → 0.0011869078874588012，
    差 **1.80 倍**，两边都逐位可复现。
    """

    def test_mismatch_is_refused_and_names_both_numbers(self):
        problems = ra.check_probed_threads(16, 24)
        assert problems
        assert "16" in problems[0] and "24" in problems[0]
        assert "1.80" in problems[0], "证据要跟着报错一起出现"

    def test_match_passes(self):
        assert ra.check_probed_threads(1, 1) == []

    def test_unknown_probe_is_not_a_failure(self):
        """探不出线程数时不判——假装检查过比不检查更糟。"""
        assert ra.check_probed_threads(None, 24) == []

    def test_probe_returns_none_for_a_bogus_interpreter(self):
        assert ra.probe_thread_count("/nonexistent/python-xyz", 1) is None


class TestPinnedEnv:
    def test_all_four_thread_variables_are_pinned(self):
        env = ra.pinned_env(3)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            assert env[key] == "3"

    def test_threads_one_is_not_treated_as_unset(self):
        # 1 是默认值，也是"关掉并行"的合法取值。用 `or` 兜底会把它换成空串。
        assert ra.pinned_env(1)["OMP_NUM_THREADS"] == "1"

    def test_pythonpath_defaults_to_src(self):
        env = ra.pinned_env(1)
        assert "src" in env.get("PYTHONPATH", "")


# ---------------------------------------------------------------- 产物与读入


class TestLoadRuns:
    def test_partition_files_are_not_runs(self, tmp_path):
        (tmp_path / "partition-mvp-k5-iid.json").write_text(
            json.dumps({"partition": {"a": ["b"]}}), encoding="utf-8"
        )
        (tmp_path / "c0__centralized__seed0.json").write_text(
            json.dumps({"run_id": "c0__centralized__seed0", "final_mae": 0.1}),
            encoding="utf-8",
        )
        runs = ra.load_runs(tmp_path)
        assert list(runs) == ["c0__centralized__seed0"]

    def test_run_id_falls_back_to_filename(self, tmp_path):
        (tmp_path / "x.json").write_text(json.dumps({"final_mae": 0.1}), encoding="utf-8")
        assert list(ra.load_runs(tmp_path)) == ["x"]

    def test_broken_json_is_not_a_run(self, tmp_path):
        (tmp_path / "bad.json").write_text("{ 不是 json", encoding="utf-8")
        assert ra.load_runs(tmp_path) == {}


class TestVerifyOutputs:
    def _write(self, path: Path, run_id: str, *, with_mae: bool = True) -> None:
        payload = {"run_id": run_id}
        if with_mae:
            payload["final_mae"] = 0.1
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_all_present_passes(self, tmp_path):
        self._write(tmp_path / "a.json", "a")
        assert ra.verify_outputs({"a": {}}, tmp_path) == []

    def test_missing_file_is_reported(self, tmp_path):
        problems = ra.verify_outputs({"a": {}}, tmp_path)
        assert problems and "缺少产物" in problems[0]

    def test_missing_final_mae_is_reported(self, tmp_path):
        self._write(tmp_path / "a.json", "a", with_mae=False)
        problems = ra.verify_outputs({"a": {}}, tmp_path)
        assert problems and "final_mae" in problems[0]

    def test_run_id_mismatch_is_reported(self, tmp_path):
        # 文件名对但内部 run_id 不对：这是"复制了一份结果冒充另一个 run"的形态。
        self._write(tmp_path / "a.json", "b")
        problems = ra.verify_outputs({"a": {}}, tmp_path)
        assert problems and "不一致" in problems[0]


class TestMixedDirectory:
    """「部分有字段、部分没有」是另一种错误，必须与「全都没有」分开。

    全都没有 = 一整批旧结果（合法）；部分有 = 新旧混在一个目录里（危险）。
    原来的检查只看「有字段的那些」，于是混杂目录会得到 {1} 并判定一致，
    被安静放行。
    """

    def test_mixed_directory_is_refused(self):
        runs = {
            "new": {"torch_num_threads": 1},
            "old": {"torch_version": "2.10.0+cpu"},
        }
        problems = ra.check_thread_count(runs, 1)
        assert problems, "混杂目录必须被拒绝"
        assert "1/2" in problems[0]
        assert "old" in problems[0]

    def test_all_old_is_still_tolerated(self):
        runs = {"a": {"torch_version": "2.10.0+cpu"},
                "b": {"torch_version": "2.10.0+cpu"}}
        assert ra.check_thread_count(runs, 1) == []


class TestSkipExistingFreshness:
    """`--skip-existing` 只能跳「本次设置」的产物。

    原来只判文件存不存在。`--out` 默认是 `results/`，那里装的可能是上一版的
    结果，于是重跑中断后续跑会跳过全部任务、打印「剩 0 个」、退出码 0，接着
    汇总出一张新旧混杂的表——不抛异常、行数也对，只有数字是旧的。
    """

    def _write(self, path, **fields):
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"final_mae": 0.001, **fields}
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_missing_file_is_not_current(self, tmp_path):
        assert not ra.artifact_is_current(
            tmp_path / "x.json",
            torch_version="2.10.0+cpu",
            threads=1,
            feature_version="discharge-v2",
        )

    def test_old_artifact_without_thread_field_is_not_current(self, tmp_path):
        """旧产物没有 torch_num_threads，必须重跑而不是跳过。"""
        path = self._write(tmp_path / "x.json", torch_version="2.10.0+cpu")
        assert not ra.artifact_is_current(
            path, torch_version="2.10.0+cpu", threads=1, feature_version=None
        )

    def test_other_thread_count_is_not_current(self, tmp_path):
        path = self._write(
            tmp_path / "x.json", torch_version="2.10.0+cpu",
            torch_num_threads=24, feature_version="discharge-v2",
        )
        assert not ra.artifact_is_current(
            path, torch_version="2.10.0+cpu", threads=1,
            feature_version="discharge-v2",
        )

    def test_other_feature_version_is_not_current(self, tmp_path):
        path = self._write(
            tmp_path / "x.json", torch_version="2.10.0+cpu",
            torch_num_threads=1, feature_version="charge-v1",
        )
        assert not ra.artifact_is_current(
            path, torch_version="2.10.0+cpu", threads=1,
            feature_version="discharge-v2",
        )

    def test_other_torch_version_is_not_current(self, tmp_path):
        path = self._write(
            tmp_path / "x.json", torch_version="2.14.1+cpu",
            torch_num_threads=1, feature_version="discharge-v2",
        )
        assert not ra.artifact_is_current(
            path, torch_version="2.10.0+cpu", threads=1,
            feature_version="discharge-v2",
        )

    def test_matching_artifact_is_current(self, tmp_path):
        path = self._write(
            tmp_path / "x.json", torch_version="2.10.0+cpu",
            torch_num_threads=1, feature_version="discharge-v2",
        )
        assert ra.artifact_is_current(
            path, torch_version="2.10.0+cpu", threads=1,
            feature_version="discharge-v2",
        )

    def test_unknown_expectation_does_not_veto_but_threads_still_count(self, tmp_path):
        """期望值读不出就不拿它下结论；线程数照样要查。"""
        ok = self._write(tmp_path / "ok.json", torch_num_threads=1)
        assert ra.artifact_is_current(
            ok, torch_version=None, threads=1, feature_version=None
        )
        bad = self._write(tmp_path / "bad.json", torch_num_threads=24)
        assert not ra.artifact_is_current(
            bad, torch_version=None, threads=1, feature_version=None
        )

    def test_broken_json_is_not_current(self, tmp_path):
        path = tmp_path / "x.json"
        path.write_text("{not json", encoding="utf-8")
        assert not ra.artifact_is_current(
            path, torch_version=None, threads=1, feature_version=None
        )

    def test_file_without_final_mae_is_not_current(self, tmp_path):
        """划分文件也放在同一个目录，不能被当成已完成产物。"""
        path = tmp_path / "x.json"
        path.write_text('{"torch_num_threads": 1}', encoding="utf-8")
        assert not ra.artifact_is_current(
            path, torch_version=None, threads=1, feature_version=None
        )


class TestMixedDirectoryResume:
    """混杂目录在 `--skip-existing` 下必须放行，否则中断一次就续不上。

    上一版把「部分有字段」一律判失败。那条规则单独看是对的（混杂目录很危险），
    但它把**续跑的正常中间态**也判死了：重跑中断后 `results/` 必然半新半旧，
    于是用户只能删掉已完成的一半重来——而删数据比混数据更贵。

    放行的前提是 `--skip-existing`：那时每个产物都会过 `artifact_is_current`
    逐个核对，旧产物线程数对不上就重新排队，混杂不再是静默的。
    """

    def test_mixed_is_refused_without_skip_existing(self):
        runs = {"new": {"torch_num_threads": 1}, "old": {}}
        assert ra.check_thread_count(runs, 1), "不带 --skip-existing 时必须拒绝"

    def test_mixed_is_allowed_with_skip_existing(self):
        runs = {"new": {"torch_num_threads": 1}, "old": {}}
        assert ra.check_thread_count(runs, 1, allow_mixed=True) == []

    def test_allow_mixed_does_not_disable_the_value_check(self):
        """放行「混杂」不等于放行「线程数不对」——那是两件事。"""
        runs = {"a": {"torch_num_threads": 24}, "b": {}}
        problems = ra.check_thread_count(runs, 1, allow_mixed=True)
        assert problems and "24" in problems[0]
