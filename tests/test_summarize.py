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
        assert res["n_pos"] == 0 and res["n_neg"] == 0
        assert res["p_t"] == pytest.approx(1.0)

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

    `flsoh` 是允许的——但只有 `flsoh.stats`，它本身也是纯标准库
    （见 tests/test_stats.py，那边不 import 任何第三方包）。
    """
    allowed = {
        "argparse", "json", "statistics", "sys", "pathlib",
        "__future__", "typing", "collections", "math", "flsoh",
    }
    src = Path(sr.__file__).read_text(encoding="utf-8")
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("import "):
            mod = s.split()[1].split(".")[0].rstrip(",")
            assert mod in allowed, f"多出了非标准库依赖：{mod}"
        if s.startswith("from ") and " import " in s:
            mod = s.split()[1].split(".")[0]
            assert mod in allowed, f"多出了非标准库依赖：{mod}"


def test_stats_module_is_stdlib_only():
    """判定逻辑的依赖面同样必须是零。"""
    import flsoh.stats as st

    allowed = {"__future__", "math"}
    for line in Path(st.__file__).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("import "):
            assert s.split()[1].split(".")[0].rstrip(",") in allowed
        if s.startswith("from ") and " import " in s:
            assert s.split()[1].split(".")[0] in allowed


class TestJudge:
    """H1/H2/H3 的判定必须机械可复现。

    判据原来靠人读「−20.9% 算不算超过 10%」。人读到哪一步都可能说服自己，
    所以判成 accept 还是 reject 要由代码算出来，而不是写出来。
    """

    def _runs(self):
        runs = []
        # C0：5 个 seed，MAE 稳定在 0.0020
        for s in range(5):
            runs.append(_run("centralized", s, 0.0020, n_clients=None, epochs=300,
                             run_id=f"c0__centralized__seed{s}"))
        # IID FedAvg：每个 seed 都更好 0.0004（相对 -20%）
        for s in range(5):
            runs.append(_run("fedavg", s, 0.0016, scheme="iid", le=10,
                             run_id=f"fedavg__iid__seed{s}"))
        # Dirichlet FedAvg：每个 seed 都更差 0.0010（相对 +50%）
        for s in range(5):
            runs.append(_run("fedavg", s, 0.0030, scheme="dirichlet", alpha=0.5,
                             le=10, run_id=f"fedavg__dirichlet0.5__seed{s}"))
        # FedProx：比同划分 FedAvg 更差 0.0005
        for s in range(5):
            runs.append(_run("fedprox", s, 0.0035, scheme="dirichlet", alpha=0.5,
                             mu=0.1, le=10,
                             run_id=f"fedprox__dirichlet0.5__seed{s}"))
        return runs

    def test_h1_rejected_when_gap_exceeds_ten_percent(self):
        out = "\n".join(sr.judge_hypotheses(self._runs()))
        h1 = out.split("H2")[0]
        assert "推翻" in h1

    def test_h2_accepted_when_noniid_degrades(self):
        # 必须带上 heterogeneity.json：不提供时 manipulation check 会拦住判定
        # （这正是 TestH2ManipulationCheck 要的行为）。这里给一份「偏斜明显
        # 高于 IID」的假数据，让 H2 走到正常分档。
        hetero = {"partitions": {
            "partition-mvp-k5-iid.json":
                {"between_client_variance_fraction": 0.173},
            "partition-mvp-k5-dirichlet0.5.json":
                {"between_client_variance_fraction": 0.471},
        }}
        out = "\n".join(sr.judge_hypotheses(self._runs(), hetero=hetero))
        h2 = out.split("H2")[1].split("H3")[0]
        assert "接受" in h2

    def test_h3_rejected_when_fedprox_is_worse(self):
        out = "\n".join(sr.judge_hypotheses(self._runs()))
        h3 = out.split("H3")[1]
        assert "推翻" in h3

    def test_three_seeds_cannot_decide(self):
        """负控：只给 3 个 seed 时必须判「未判定」，不许硬判。

        n=3 时配对 t 的 df=2、临界值 4.30；把差造得再大也可能不显著。
        这里构造一组「差很大但 seed 间抖动更大」的数据，验证判定退化为未判定。
        """
        runs = []
        for s in range(3):
            runs.append(_run("centralized", s, 0.0020, n_clients=None, epochs=300,
                             run_id=f"c0__centralized__seed{s}"))
        # 差 −0.0002 / −0.0004 / −0.0006：均值 −0.0004，sd 0.0002 → t=3.46 < 4.30
        for s, mae in enumerate([0.0018, 0.0016, 0.0014]):
            runs.append(_run("fedavg", s, mae, scheme="iid", le=10,
                             run_id=f"fedavg__iid__seed{s}"))
        out = "\n".join(sr.judge_hypotheses(runs))
        assert "未判定" in out.split("H2")[0]

    def test_missing_baseline_is_reported(self):
        runs = [r for r in self._runs() if r["strategy"] != "centralized"]
        out = "\n".join(sr.judge_hypotheses(runs))
        assert "C0 缺失" in out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


class TestH2ManipulationCheck:
    """判 H2 之前必须先确认处理真的被施加了。

    背景：2026-10-24 实测，Dirichlet α=0.5 在 N=23 时造出的标签偏斜是
    12.2%，**低于 IID 的 17.3%**——因为一个大客户端拿了 21 颗电池，把电池
    间的 SOH 差异平均掉了。所以「Non-IID 没掉点」有两种解释：算法免疫，
    或者处理没施加。MAE 表上这两种解释一模一样，而结论相反。
    判定必须选后者（未判定），不能选前者。

    对照组是**同 local_epochs 的 IID FedAvg**：H2 问的是划分方式造成多少
    损失，不是联邦与中心化的差距（那是 H1）。所以 fixture 里必须有 IID 组。
    """

    @staticmethod
    def _hetero(dir_skew: float, iid_skew: float, rank_skew: float | None = None):
        parts = {
            "partition-mvp-k5-iid.json": {
                "between_client_variance_fraction": iid_skew
            },
            "partition-mvp-k5-dirichlet0.5.json": {
                "between_client_variance_fraction": dir_skew
            },
        }
        if rank_skew is not None:
            parts["partition-mvp-k5-sohrank.json"] = {
                "between_client_variance_fraction": rank_skew
            }
        return {"partitions": parts}

    @staticmethod
    def _runs():
        """IID 0.0020 / Dirichlet 0.0012 / C0 0.0022。

        Dirichlet 比 IID 好 40%——若不做 manipulation check 会被判成
        「H2 推翻（Non-IID 没掉点）」。做了检查才知道它其实更均匀。
        """
        runs = []
        for s, mae in enumerate([0.0022, 0.0024, 0.0020]):
            runs.append(_run("centralized", s, mae, n_clients=None,
                             epochs=300, run_id=f"c0__centralized__seed{s}"))
        for s, mae in enumerate([0.0020, 0.0022, 0.0018]):
            runs.append(_run("fedavg", s, mae, scheme="iid", le=10,
                             run_id=f"fedavg__iid__seed{s}"))
        for s, mae in enumerate([0.0012, 0.0011, 0.0013]):
            runs.append(_run("fedavg", s, mae, scheme="dirichlet", le=10,
                             run_id=f"fedavg__dirichlet0.5__seed{s}"))
        return runs

    def test_skew_below_iid_blocks_the_verdict(self):
        """偏斜 ≤ IID 基线时，不许判「H2 推翻」。"""
        runs = self._runs()
        hetero = self._hetero(dir_skew=0.122, iid_skew=0.173)
        out = "\n".join(sr.judge_hypotheses(runs, hetero=hetero))
        h2 = out.split("H2")[1].split("H3")[0]
        assert "处理未施加" in h2
        assert "推翻" not in h2, "处理没施加却判了推翻"

    def test_skew_above_iid_allows_the_verdict(self):
        """偏斜 > IID 基线时，正常按阈值判。"""
        runs = self._runs()
        hetero = self._hetero(dir_skew=0.471, iid_skew=0.173)
        out = "\n".join(sr.judge_hypotheses(runs, hetero=hetero))
        h2 = out.split("H2")[1].split("H3")[0]
        assert "处理未施加" not in h2
        # Dirichlet 显著好于 C0 → 推翻 H2
        assert "推翻" in h2

    def test_missing_hetero_blocks_the_verdict(self):
        """没有 heterogeneity.json 时不许判——不静默当成「异质性没问题」。"""
        runs = self._runs()
        out = "\n".join(sr.judge_hypotheses(runs, hetero={}))
        h2 = out.split("H2")[1].split("H3")[0]
        assert "未判定" in h2
        assert "推翻" not in h2

    def test_sohrank_reports_its_skew_inline(self):
        """偏斜分片要在输出里带上偏斜数值，读者才知道它在测什么。"""
        runs = self._runs()
        for s, mae in enumerate([0.0026, 0.0028, 0.0024]):
            runs.append(_run("fedavg", s, mae, scheme="sohrank", le=10,
                                run_id=f"fedavg__sohrank__seed{s}"))
        hetero = self._hetero(0.122, 0.173, rank_skew=0.471)
        out = "\n".join(sr.judge_hypotheses(runs, hetero=hetero))
        assert "47.1%" in out
        assert "17.3%" in out


class TestH2ControlIsIidNotC0:
    """H2 的对照组是同预算的 IID FedAvg，不是 C0。

    理由：H2 问「划分方式造成多少损失」。拿 C0 当分母会把「联邦 vs 中心化」
    （H1）和「IID vs Non-IID」（H2）两个效应混在一个数里。
    实测两者给出相反的画面：Dirichlet 相对 C0 是 −35.9%（好很多），
    相对同预算 IID 是 −16.3%（还是好，但幅度差一半）。
    """

    def test_verdict_cites_iid_not_c0(self):
        runs = TestH2ManipulationCheck._runs()
        hetero = TestH2ManipulationCheck._hetero(0.471, 0.173, rank_skew=0.471)
        for s, mae in enumerate([0.0026, 0.0028, 0.0024]):
            runs.append(_run("fedavg", s, mae, scheme="sohrank", le=10,
                             run_id=f"fedavg__sohrank__seed{s}"))
        out = "\n".join(sr.judge_hypotheses(runs, hetero=hetero))
        h2 = out.split("H2")[1].split("H3")[0]
        assert "相对同预算 IID" in h2
        assert "相对 C0" not in h2, "H2 不应该拿 C0 当分母"

    def test_missing_iid_control_is_reported(self):
        """没有同预算 IID 组时必须明说，不能默默退回 C0。"""
        runs = []
        for s, mae in enumerate([0.0022, 0.0024, 0.0020]):
            runs.append(_run("centralized", s, mae, n_clients=None,
                             epochs=300, run_id=f"c0__centralized__seed{s}"))
        for s, mae in enumerate([0.0026, 0.0028, 0.0024]):
            runs.append(_run("fedavg", s, mae, scheme="sohrank", le=10,
                             run_id=f"fedavg__sohrank__seed{s}"))
        out = "\n".join(sr.judge_hypotheses(runs, hetero={}))
        h2 = out.split("H2")[1].split("H3")[0]
        assert "缺同预算" in h2
        assert "未判定" not in h2 or "缺同预算" in h2

    def test_different_local_epochs_are_separate_controls(self):
        """le=3 的 Non-IID 不能拿 le=10 的 IID 当对照组。

        预算不同就不是同一个实验，配对出来的差会被训练量差异污染。
        """
        runs = TestH2ManipulationCheck._runs()   # IID le=10
        for s, mae in enumerate([0.0040, 0.0042, 0.0038]):
            runs.append(_run("fedavg", s, mae, scheme="sohrank", le=3,
                             run_id=f"fedavg__sohrank__le3__seed{s}"))
        out = "\n".join(sr.judge_hypotheses(
            runs, hetero=TestH2ManipulationCheck._hetero(0.122, 0.173, 0.471)
        ))
        h2 = out.split("H2")[1].split("H3")[0]
        assert "缺同预算（local_epochs=3）的 IID 对照组" in h2
