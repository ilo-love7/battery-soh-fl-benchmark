"""`scripts/promote_results.py`：提升重跑产物时不许「提升一半」。

为什么值得单独测
----------------
这个脚本只做一件事——把重跑目录里的 JSON 拷进正式结果目录。而它能造成的
唯一一种伤害是**静默混杂**：少拷几个、或拷进口径不同的几个，于是
`results/` 里新旧混在一起。

混杂之后 `summarize_results.py --write-plan docs/plan.md` 会把混杂的表写进
文档，而 CI 的 `--check-plan` 会**通过**——它只核对「文档与 results/ 是否
一致」，不核对「results/ 内部是否同源」。一张永久写错的头版表，闸门全绿。

所以下面每条测试都对应一种「提升了一半」的具体走法。
"""

from __future__ import annotations

import json

import pytest

pr = pytest.importorskip("promote_results")


def _write(directory, run_id, **fields):
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"final_mae": 0.001, **fields}
    path = directory / f"{run_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


GOOD = {"feature_version": "discharge-v2", "torch_num_threads": 1}


class TestCheckSource:
    def test_matching_source_passes(self):
        runs = {"a": dict(GOOD), "b": dict(GOOD)}
        assert pr.check_source(
            runs, expect_feature_version="discharge-v2", threads=1
        ) == []

    def test_stale_feature_version_is_refused(self):
        """源目录里混进上一版产物 → 提升上去就是新旧混杂。"""
        runs = {"a": dict(GOOD), "old": {"feature_version": None,
                                         "torch_num_threads": 1}}
        problems = pr.check_source(
            runs, expect_feature_version="discharge-v2", threads=1
        )
        assert problems and "old" in problems[0]

    def test_wrong_thread_count_is_refused(self):
        runs = {"a": dict(GOOD), "b": {"feature_version": "discharge-v2",
                                       "torch_num_threads": 16}}
        problems = pr.check_source(
            runs, expect_feature_version="discharge-v2", threads=1
        )
        assert problems and "16" in problems[0]

    def test_unknown_expectation_is_not_checked(self):
        """读不出的期望值不拿来下结论——但调用点必须知道「没检查」。"""
        runs = {"a": {"feature_version": None, "torch_num_threads": None}}
        assert pr.check_source(
            runs, expect_feature_version=None, threads=None
        ) == []


class TestCheckCoverage:
    def test_subset_passes(self):
        source = {"a": dict(GOOD), "b": dict(GOOD)}
        target = {"a": dict(GOOD)}
        assert pr.check_coverage(source, target) == []

    def test_target_only_run_is_refused(self):
        """目标里有一个源里没有的 run → 提升后那个位置还是旧结果。

        这是最隐蔽的一种混杂：文件数没变、每个 JSON 都合法、seeds 也齐，
        出错的是**集合**而不是数值。
        """
        source = {"a": dict(GOOD)}
        target = {"a": dict(GOOD), "leftover": dict(GOOD)}
        problems = pr.check_coverage(source, target)
        assert problems and "leftover" in problems[0]


class TestCheckAfter:
    def test_clean_target_passes(self):
        target = {"a": dict(GOOD)}
        assert pr.check_after(target, expect_feature_version="discharge-v2") == []

    def test_remaining_old_run_is_refused(self):
        """拷完之后仍有一个口径不符的 run → 兜底必须报出来。"""
        target = {"a": dict(GOOD), "b": {"feature_version": None}}
        problems = pr.check_after(target, expect_feature_version="discharge-v2")
        assert problems and "b" in problems[0]


class TestMain:
    def test_dry_run_copies_nothing(self, tmp_path):
        src, dst = tmp_path / "src", tmp_path / "dst"
        _write(src, "a", **GOOD)
        _write(dst, "a", feature_version=None, torch_num_threads=16)
        code = pr.main(["--from", str(src), "--to", str(dst),
                        "--expect-feature-version", "discharge-v2",
                        "--threads", "1"])
        assert code == 0
        assert json.loads((dst / "a.json").read_text())["feature_version"] is None

    def test_execute_replaces_and_leaves_partition_files_alone(self, tmp_path):
        src, dst = tmp_path / "src", tmp_path / "dst"
        _write(src, "a", **GOOD)
        _write(dst, "a", feature_version=None, torch_num_threads=16)
        # 划分文件没有 final_mae，不属于 run，必须原地不动
        (dst / "partition-mvp-k5-iid.json").write_text(
            json.dumps({"clients": [[0, 1]]}), encoding="utf-8")
        code = pr.main(["--from", str(src), "--to", str(dst),
                        "--expect-feature-version", "discharge-v2",
                        "--threads", "1", "--execute"])
        assert code == 0
        assert json.loads((dst / "a.json").read_text())["feature_version"] == "discharge-v2"
        assert (dst / "partition-mvp-k5-iid.json").is_file()

    def test_execute_refuses_and_copies_nothing_when_a_run_is_missing(self, tmp_path):
        """少一个 run 就整批不拷——「提升一半」比「不提升」更糟。"""
        src, dst = tmp_path / "src", tmp_path / "dst"
        _write(src, "a", **GOOD)
        _write(dst, "a", feature_version=None, torch_num_threads=16)
        _write(dst, "leftover", feature_version=None, torch_num_threads=16)
        code = pr.main(["--from", str(src), "--to", str(dst),
                        "--expect-feature-version", "discharge-v2",
                        "--threads", "1", "--execute"])
        assert code == 2
        assert json.loads((dst / "a.json").read_text())["feature_version"] is None

    def test_empty_source_fails(self, tmp_path):
        src, dst = tmp_path / "src", tmp_path / "dst"
        dst.mkdir(parents=True)
        assert pr.main(["--from", str(src), "--to", str(dst)]) == 2
