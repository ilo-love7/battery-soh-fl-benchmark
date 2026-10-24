"""表 1 生成器的测试。

为什么值得单独测：这张表是 CI 反造假闸门的**基准**。如果生成器本身算错，
闸门会忠实地把错的表固化下来——「文档与 results/ 一致」变成「文档与错
的算法一致」，看起来还是绿的。所以生成器的行为必须自己被钉住。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import summarize_results as sr


def _run(strategy, seed, mae, *, scheme=None, alpha=None, mu=None,
         n_clients=23, rmse=None, le=None, epochs=None, run_id=None):
    d = {
        "strategy": strategy,
        "scheme": scheme,
        "alpha": alpha,
        "mu": mu,
        "n_clients": n_clients,
        "seed": seed,
        "final_mae": mae,
        "final_rmse": rmse if rmse is not None else mae * 1.5,
        "run_id": run_id or f"{strategy}__{scheme or 'x'}__seed{seed}",
    }
    if le is not None:
        d["local_epochs"] = le
    if epochs is not None:
        d["epochs"] = epochs
    return d


def _runs():
    return [
        _run("centralized", 0, 0.0020, n_clients=None, epochs=300,
             run_id="c0__centralized__seed0"),
        _run("centralized", 1, 0.0022, n_clients=None, epochs=300,
             run_id="c0__centralized__seed1"),
        _run("local", 0, 0.0400, epochs=300, run_id="c1__local__seed0"),
        _run("fedavg", 0, 0.0016, scheme="iid", le=10,
             run_id="fedavg__iid__seed0"),
        _run("fedavg", 1, 0.0018, scheme="iid", le=10,
             run_id="fedavg__iid__seed1"),
        _run("fedavg", 0, 0.0010, scheme="dirichlet", alpha=0.5, le=10,
             run_id="fedavg__dirichlet0.5__seed0"),
        _run("fedprox", 0, 0.0011, scheme="dirichlet", alpha=0.5, mu=0.1, le=10,
             run_id="fedprox__dirichlet0.5__seed0"),
        _run("fedprox", 0, 0.0012, scheme="dirichlet", alpha=0.5, mu=0.01, le=10,
             run_id="fedprox-mu0.01-dirichlet0.5-seed0"),
    ]


class TestGrouping:
    def test_groups_by_fields_not_run_id(self):
        """改 run_id 后缀不能改变分组。

        分组若依赖 run_id 前缀，任何人给文件改个名字就会让两批 run 静默
        合并或拆开——而表看起来仍然正常。所以分组键必须是 JSON 里的字段。
        """
        runs = _runs()
        renamed = [dict(r, run_id="whatever-" + str(i))
                   for i, r in enumerate(runs)]
        assert len(sr._group(runs)) == len(sr._group(renamed))

    def test_fedprox_mu_splits_groups(self):
        groups = sr._group(_runs())
        fedprox = [k for k in groups if k[0] == "fedprox"]
        assert len(fedprox) == 2
        assert {k[3] for k in fedprox} == {0.1, 0.01}

    def test_epochs_and_local_epochs_share_the_key_slot(self):
        """C0 用 `epochs`、联邦用 `local_epochs`，两者不能混成一格。"""
        groups = sr._group(_runs())
        c0 = [k for k in groups if k[0] == "centralized"]
        assert len(c0) == 1
        assert c0[0][4] == 300


class TestCodes:
    def test_fixed_codes_match_plan(self):
        codes = sr._codes(list(sr._group(_runs())))
        by_strategy = {}
        for key, code in codes.items():
            by_strategy.setdefault(key[0], set()).add(code)
        assert by_strategy["centralized"] == {"C0"}
        assert by_strategy["local"] == {"C1"}
        assert by_strategy["fedavg"] == {"I1", "N1"}

    def test_colliding_code_gets_mu_suffix(self):
        """两个 μ 都叫 N2 的话，读者分不清哪行对应哪个 μ。"""
        codes = sr._codes(list(sr._group(_runs())))
        fedprox = sorted(c for k, c in codes.items() if k[0] == "fedprox")
        assert fedprox == ["N2@μ=0.01", "N2@μ=0.1"]


class TestBuildTable:
    def test_baseline_row_has_no_relative(self):
        rows = sr.build_table(_runs())
        c0 = [r for r in rows if r.startswith("| C0 |")]
        assert len(c0) == 1
        assert c0[0].rstrip().endswith("| — |")

    def test_relative_is_computed_against_c0_mean(self):
        rows = sr.build_table(_runs())
        # C0 均值 = 0.0021，FedAvg/IID 均值 = 0.0017 → -19.0%
        i1 = [r for r in rows if r.startswith("| I1 |")][0]
        assert "-19.0%" in i1

    def test_missing_baseline_is_fatal(self):
        """C0 缺失时必须报错，不能悄悄换分母。"""
        runs = [r for r in _runs() if r["strategy"] != "centralized"]
        with pytest.raises(SystemExit):
            sr.build_table(runs)

    def test_table_shape(self):
        rows = sr.build_table(_runs())
        header, sep = rows[0], rows[1]
        assert header.count("|") == sep.count("|")
        for r in rows:
            assert r.count("|") == header.count("|"), r


class TestCheckPlan:
    def _write(self, tmp_path: Path, table: str) -> Path:
        p = tmp_path / "plan.md"
        p.write_text(
            "# x\n\n" + sr.BEGIN_MARK + "\n" + table + "\n" + sr.END_MARK + "\n",
            encoding="utf-8",
        )
        return p

    def _dump(self, tmp_path: Path, runs) -> None:
        (tmp_path / "results").mkdir()
        for i, r in enumerate(runs):
            (tmp_path / "results" / f"r{i}.json").write_text(
                json.dumps(r), encoding="utf-8"
            )

    def test_matching_table_passes(self, tmp_path, monkeypatch):
        runs = _runs()
        self._dump(tmp_path, runs)
        plan = self._write(tmp_path, "\n".join(sr.build_table(runs)))
        assert sr.main(["--results", str(tmp_path / "results"),
                        "--check-plan", str(plan)]) == 0

    def test_one_changed_digit_fails(self, tmp_path):
        """负控：手改一个数字必须红。这是闸门存在的全部理由。"""
        runs = _runs()
        self._dump(tmp_path, runs)
        table = "\n".join(sr.build_table(runs))
        assert "0.00200" in table or "0.0017" in table
        bad = table.replace("0.0017", "0.0019", 1)
        assert bad != table
        plan = self._write(tmp_path, bad)
        assert sr.main(["--results", str(tmp_path / "results"),
                        "--check-plan", str(plan)]) == 3

    def test_missing_markers_is_fatal(self, tmp_path):
        runs = _runs()
        self._dump(tmp_path, runs)
        p = tmp_path / "plan.md"
        p.write_text("# 没有标记\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            sr.main(["--results", str(tmp_path / "results"),
                     "--check-plan", str(p)])

    def test_empty_results_dir_fails(self, tmp_path):
        (tmp_path / "results").mkdir()
        plan = self._write(tmp_path, "| a |")
        assert sr.main(["--results", str(tmp_path / "results"),
                        "--check-plan", str(plan)]) == 1


class TestPaired:
    def test_identical_groups_have_zero_difference(self):
        runs = _runs()
        a = [r for r in runs if r["strategy"] == "fedavg" and r["scheme"] == "iid"]
        res = sr._paired(a, a)
        assert res["mean"] == 0.0
        assert res["stable"] is True

    def test_pairs_on_common_seeds_only(self):
        a = [_run("fedavg", 0, 0.0010, scheme="iid", le=10),
             _run("fedavg", 1, 0.0012, scheme="iid", le=10)]
        b = [_run("centralized", 1, 0.0020, epochs=300, n_clients=None),
             _run("centralized", 2, 0.0030, epochs=300, n_clients=None)]
        res = sr._paired(a, b)
        # 只有 seed=1 共同 → 不足 2 个，返回 None 而不是硬算
        assert res is None

    def test_too_few_common_seeds_returns_none(self):
        a = [_run("fedavg", 0, 0.0010, scheme="iid", le=10)]
        b = [_run("centralized", 0, 0.0020, epochs=300, n_clients=None)]
        assert sr._paired(a, b) is None


def test_script_uses_only_stdlib():
    """CI 的 no-fabricated-results job 只装纯 Python。

    这个脚本一旦 import 了 torch/numpy，那道闸门就会在 import 阶段崩，
    而不是给出「表不一致」这种可操作的报错。
    """
    allowed = {
        "argparse", "json", "statistics", "sys", "pathlib",
        "__future__", "typing", "collections",
    }
    src = Path(sr.__file__).read_text(encoding="utf-8")
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("import ") and not s.startswith("import from"):
            mod = s.split()[1].split(".")[0].rstrip(",")
            assert mod in allowed, f"多出了非标准库依赖：{mod}"
        if s.startswith("from ") and " import " in s:
            mod = s.split()[1].split(".")[0]
            assert mod in allowed, f"多出了非标准库依赖：{mod}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
