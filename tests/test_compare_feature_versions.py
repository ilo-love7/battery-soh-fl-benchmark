"""新旧口径对比脚本的测试。

这个脚本的产出会直接进文档（「特征修复改了多少」），所以它自己必须被钉住。
三处最容易悄悄错的地方：

1. **符号检验的 p 值**。n=8 全同向时 p=0.0078，这是判定「方向可数」的最低门槛
   （见 ``docs/plan.md`` §11.3.1）。算成单侧、算成 0.0039、或者把平局算进 n，
   都会让一个不显著的方向看起来显著。
2. **线程数不同的两个 run 被配对**。那样得到的比值混着「线程数」这个与 seed
   同量级的随机源，数值照样漂亮，含义为零。必须拒绝对比，不是提醒。
3. **配对太少还给结论**。3 对全同向的 p 是 0.25。
"""

from __future__ import annotations

import json

import pytest

import compare_feature_versions as cfv


def _write(directory, run_id, mae, *, threads=1):
    payload = {"run_id": run_id, "final_mae": mae}
    if threads is not None:
        payload["torch_num_threads"] = threads
    path = directory / f"{run_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestSignTest:
    def test_all_same_direction_n8_is_the_documented_floor(self):
        """n=8 全同向 → 2/256 = 0.0078125。文档里引用的就是这个数。"""
        assert cfv.sign_test_p(8, 0) == pytest.approx(0.0078125)

    def test_is_symmetric(self):
        assert cfv.sign_test_p(7, 17) == pytest.approx(cfv.sign_test_p(17, 7))

    def test_even_split_is_one(self):
        assert cfv.sign_test_p(4, 4) == pytest.approx(1.0)

    def test_no_pairs_is_one(self):
        assert cfv.sign_test_p(0, 0) == pytest.approx(1.0)

    def test_matches_hand_computation_for_7_of_24(self):
        """24 对里 7 好 17 差：2·Σ_{i≤7} C(24,i) / 2²⁴ = 0.0639。"""
        assert cfv.sign_test_p(7, 17) == pytest.approx(0.0639, abs=1e-4)

    def test_never_exceeds_one(self):
        for better in range(0, 6):
            for worse in range(0, 6):
                assert 0.0 < cfv.sign_test_p(better, worse) <= 1.0


class TestFamilyOf:
    def test_strips_double_underscore_seed(self):
        assert cfv.family_of("fedavg__iid__le3__nm0__seed3") == "fedavg__iid__le3__nm0"

    def test_strips_dash_seed(self):
        assert cfv.family_of("fedavg-le3-iid-seed3") == "fedavg-le3-iid"

    def test_leaves_seedless_ids_alone(self):
        assert cfv.family_of("c0__centralized") == "c0__centralized"

    def test_single_seed_suffix_does_not_become_empty(self):
        """``seed3`` 这种以 seed 开头的 id 不能被切空。"""
        assert cfv.family_of("seed3") == "seed3"


class TestQuantiles:
    def test_two_points(self):
        q1, q3 = cfv._quartiles([1.0, 3.0])
        assert q1 == pytest.approx(1.5)
        assert q3 == pytest.approx(2.5)

    def test_single_point(self):
        assert cfv._quartiles([2.0]) == (2.0, 2.0)

    def test_matches_known_values(self):
        # 1..5 的 25%/75% 分位（线性插值）：2.0 与 4.0
        q1, q3 = cfv._quartiles([1.0, 2.0, 3.0, 4.0, 5.0])
        assert q1 == pytest.approx(2.0)
        assert q3 == pytest.approx(4.0)


class TestLoadRuns:
    def test_skips_files_without_final_mae(self, tmp_path):
        """划分产物（partition-*.json）与坏 JSON 都不是一次实验的产物。"""
        _write(tmp_path, "a__seed0", 0.01)
        (tmp_path / "partition-k5.json").write_text(json.dumps({"clients": []}), encoding="utf-8")
        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        runs = cfv.load_runs(tmp_path)
        assert list(runs) == ["a__seed0"]

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cfv.load_runs(tmp_path / "nope")


class TestCli:
    def test_refuses_to_pair_different_thread_counts(self, tmp_path, capsys):
        """线程数不同就不配对——这是硬闸门，不是提醒。"""
        old, new = tmp_path / "old", tmp_path / "new"
        old.mkdir()
        new.mkdir()
        for i in range(6):
            _write(old, f"a__seed{i}", 0.010, threads=24)
            _write(new, f"a__seed{i}", 0.011, threads=1)
        code = cfv.main(["--old", str(old), "--new", str(new)])
        assert code == 2
        err = capsys.readouterr().err
        assert "少于" in err

    def test_too_few_pairs_is_refused(self, tmp_path, capsys):
        old, new = tmp_path / "old", tmp_path / "new"
        old.mkdir()
        new.mkdir()
        for i in range(3):
            _write(old, f"a__seed{i}", 0.010)
            _write(new, f"a__seed{i}", 0.011)
        assert cfv.main(["--old", str(old), "--new", str(new)]) == 2
        assert "少于" in capsys.readouterr().err

    def test_reports_median_and_sign_test(self, tmp_path, capsys):
        old, new = tmp_path / "old", tmp_path / "new"
        old.mkdir()
        new.mkdir()
        # 6 对：全部变差一倍 → 中位 2.000、p = 2/64 = 0.031
        for i in range(6):
            _write(old, f"a__seed{i}", 0.010)
            _write(new, f"a__seed{i}", 0.020)
        assert cfv.main(["--old", str(old), "--new", str(new)]) == 0
        out = capsys.readouterr().out
        assert "2.000" in out
        assert "0.0312" in out
        assert "新表更好 0 / 更差 6" in out

    def test_seed_cv_needs_three_points(self):
        runs = {"a__seed0": {"final_mae": 1.0}, "a__seed1": {"final_mae": 2.0}}
        assert cfv.seed_cv(runs, "a") is None
        runs["a__seed2"] = {"final_mae": 3.0}
        # stdev([1,2,3]) = 1.0，均值 2.0 → CV = 0.5
        assert cfv.seed_cv(runs, "a") == pytest.approx(0.5)
