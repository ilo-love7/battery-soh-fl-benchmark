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
         n_clients=23, rmse=None, le=None, epochs=None, run_id=None,
         server_lr=None, server_tau=None, server_beta_1=None,
         server_beta_2=None, noise_multiplier=None, max_grad_norm=None,
         dp_epsilon=None):
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
    if server_lr is not None:
        # 与 scripts/run_federated.py 写盘的字段一致：四个一起写，
        # 否则「只差 τ」的两个 run 在测试里也塌成一组。
        d["server_lr"] = server_lr
        d["server_beta_1"] = 0.9 if server_beta_1 is None else server_beta_1
        d["server_beta_2"] = 0.99 if server_beta_2 is None else server_beta_2
        d["server_tau"] = 1e-3 if server_tau is None else server_tau
    if noise_multiplier is not None:
        # 与 run_federated.py 一致：noise_multiplier 一写就连带 max_grad_norm
        # 与 dp_epsilon。只写 nm 会让「同 nm 不同裁剪范数」的两个 run 在
        # 测试里也塌成一组——正是 τ 那年踩过的坑。
        d["noise_multiplier"] = noise_multiplier
        d["max_grad_norm"] = 1.0 if max_grad_norm is None else max_grad_norm
        d["target_delta"] = 1e-5
        d["dp_epsilon"] = dp_epsilon
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


class TestSeedCvAdvice:
    """CV 超标时的提示不能与 plan §11.3.1 的结论打架。

    上一版的提示是「超标项需要加 seed 数」，而同一段输出紧接着就打印配对
    比较。plan §11.3.1 已经算过加 seed 不划算（CV 16% 想分辨 5% 要 ~50 个，
    4 配置 × 50 seed = 10–20 小时），最终选的是配对比较。按旧提示读，读者
    会把一份已完成的判定当成待办。
    """

    def test_no_advice_when_all_within_threshold(self):
        assert sr.seed_cv_advice([]) is None

    def test_advice_points_to_paired_comparison(self):
        text = sr.seed_cv_advice([("A C0 中心化", 14.3)])
        assert "配对比较" in text
        # 旧文案的断言式措辞必须消失：它把可选路径写成了强制动作
        assert "需要加 seed 数" not in text

    def test_advice_names_the_worst_offender(self):
        text = sr.seed_cv_advice([("A C0", 14.3), ("D I1", 66.1), ("B C1", 28.0)])
        assert "66.1%" in text and "D I1" in text
        assert "3 个配置" in text

    def test_report_flags_only_groups_over_ten_percent(self):
        runs = [
            # 稳定：0.0020 ± 0.00002 → CV ~1%
            _run("centralized", s, 0.0020 + 0.00002 * (s % 3),
                 n_clients=None, epochs=300, run_id=f"c0__centralized__seed{s}")
            for s in range(3)
        ] + [
            # 抖动：0.0010 / 0.0016 / 0.0022 → CV ~37%
            _run("fedavg", s, 0.0010 + 0.0006 * s, scheme="iid", le=3,
                 run_id=f"fedavg__iid__seed{s}")
            for s in range(3)
        ]
        groups = sr._group(runs)
        codes = sr._codes(list(groups))
        lines, bad = sr.seed_cv_report(groups, codes, lambda k: f"{codes[k]}")
        assert len(lines) == 2
        # 只有抖动那一组超标，且 CV 确实是 37% 那个量级（不是随手判的）
        assert len(bad) == 1
        assert bad[0][1] == pytest.approx(37.0, abs=2.0)
        assert "只有 1 个 seed" not in "\n".join(lines)

    def test_single_seed_group_is_not_judged(self):
        runs = [_run("fedavg", 0, 0.0010, scheme="iid", le=3,
                     run_id="fedavg__iid__seed0")]
        groups = sr._group(runs)
        codes = sr._codes(list(groups))
        lines, bad = sr.seed_cv_report(groups, codes, lambda k: f"{codes[k]}")
        assert bad == []
        assert "只有 1 个 seed" in lines[0]


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
                {"between_client_variance_fraction": 0.173, "n_clients": 23},
            "partition-mvp-k5-dirichlet0.5.json":
                {"between_client_variance_fraction": 0.471, "n_clients": 23},
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
        """heterogeneity.json 的 fixture。

        `n_clients` 必须写：判定器按 (scheme, N) 取偏斜，缺了这个字段
        就等于「找不到该 N 的划分」，manipulation check 会被跳过。
        """
        parts = {
            "partition-mvp-k5-iid.json": {
                "between_client_variance_fraction": iid_skew,
                "n_clients": 23,
            },
            "partition-mvp-k5-dirichlet0.5.json": {
                "between_client_variance_fraction": dir_skew,
                "n_clients": 23,
            },
        }
        if rank_skew is not None:
            parts["partition-mvp-k5-sohrank.json"] = {
                "between_client_variance_fraction": rank_skew,
                "n_clients": 23,
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
        assert "缺同预算（local_epochs=3、N=23）的 IID 对照组" in h2

    def test_different_n_are_separate_controls(self):
        """N 不同也不是同一个实验：k 扫描里每档都要同 N 的 IID 对照。

        sohrank k=2（N=58）不能拿 k=5 的 IID（N=23）当分母——「客户端多了
        35 个」本身就会改变每客户端数据量与梯度步数，这个效应会被算到
        偏斜头上。
        """
        runs = TestH2ManipulationCheck._runs()   # IID N=23
        for s, mae in enumerate([0.0040, 0.0042, 0.0038]):
            runs.append(_run("fedavg", s, mae, scheme="sohrank", le=10,
                             n_clients=58,
                             run_id=f"fedavg__sohrank_k2__seed{s}"))
        out = "\n".join(sr.judge_hypotheses(
            runs, hetero=TestH2ManipulationCheck._hetero(0.122, 0.173, 0.471)
        ))
        h2 = out.split("H2")[1].split("H3")[0]
        assert "缺同预算（local_epochs=10、N=58）的 IID 对照组" in h2


class TestKSweepGrouping:
    """M3 的 k 扫描：同一 (strategy, scheme, le) 会对应四个 N。

    不把 N 放进分组键的后果不是「少一行」，而是四档实验塌成一行、
    均值描述任何一个都不存在的配置——而表看起来照样正常。
    """

    @staticmethod
    def _ksweep_runs():
        runs = []
        for s, mae in enumerate([0.0022, 0.0024, 0.0020]):
            runs.append(_run("centralized", s, mae, n_clients=None,
                             epochs=300, run_id=f"c0__centralized__seed{s}"))
        for n in (12, 23, 58, 115):
            for s, mae in enumerate([0.0020, 0.0022, 0.0018]):
                runs.append(_run("fedavg", s, mae, scheme="sohrank", le=10,
                                 n_clients=n,
                                 run_id=f"fedavg__sohrank_k{n}__seed{s}"))
        return runs

    def test_n_clients_splits_groups(self):
        groups = sr._group(self._ksweep_runs())
        rank = [k for k in groups if k[0] == "fedavg"]
        assert len(rank) == 4
        assert {k[5] for k in rank} == {12, 23, 58, 115}

    def test_codes_get_n_suffix(self):
        codes = sr._codes(list(sr._group(self._ksweep_runs())))
        fedavg = {c for k, c in codes.items() if k[0] == "fedavg"}
        assert fedavg == {"P1@N=12", "P1@N=23", "P1@N=58", "P1@N=115"}

    def test_table_sorts_n_numerically(self):
        """字典序会把 N=115 排到 N=12 前面（"115" < "12"）。"""
        rows = sr.build_table(self._ksweep_runs())
        order = [r.split("|")[1].strip() for r in rows if r.startswith("| P1")]
        assert order == ["P1@N=12", "P1@N=23", "P1@N=58", "P1@N=115"]

    def test_each_row_keeps_its_own_mean(self):
        """塌成一组会把四档平均成一个数；分开后每行是自己的均值。"""
        rows = sr.build_table(self._ksweep_runs())
        assert len([r for r in rows if r.startswith("| P1")]) == 4
        for r in rows:
            if r.startswith("| P1"):
                assert "0.00200 ± 0.00020" in r

    def test_skew_lookup_is_per_n(self):
        """同一 scheme 四份划分，偏斜必须按 N 取，不能永远返回第一份。"""
        hetero = {"partitions": {
            "partition-mvp-k10-sohrank.json": {
                "between_client_variance_fraction": 0.392, "n_clients": 12},
            "partition-mvp-k5-sohrank.json": {
                "between_client_variance_fraction": 0.471, "n_clients": 23},
            "partition-mvp-k2-sohrank.json": {
                "between_client_variance_fraction": 0.572, "n_clients": 58},
            "partition-mvp-k1-sohrank.json": {
                "between_client_variance_fraction": 0.704, "n_clients": 115},
        }}
        assert sr._skew_of(hetero, "sohrank", 12) == 0.392
        assert sr._skew_of(hetero, "sohrank", 115) == 0.704
        assert sr._skew_of(hetero, "sohrank", 999) is None

    def test_equal_skew_blocks_the_verdict(self):
        """k=1 时 sohrank 与 IID 的偏斜完全相同——处理未被施加。

        一颗电池一个客户端，排序只改客户端标签、不改客户端均值的分布，
        所以这一对测出任何差异都只能是噪声。它是图 2 的内部校准点。
        """
        runs = []
        for s, mae in enumerate([0.0022, 0.0024, 0.0020]):
            runs.append(_run("centralized", s, mae, n_clients=None,
                             epochs=300, run_id=f"c0__centralized__seed{s}"))
        for s, mae in enumerate([0.0030, 0.0032, 0.0028]):
            runs.append(_run("fedavg", s, mae, scheme="iid", le=10,
                             n_clients=115,
                             run_id=f"fedavg__iid_k1__seed{s}"))
        for s, mae in enumerate([0.0031, 0.0033, 0.0029]):
            runs.append(_run("fedavg", s, mae, scheme="sohrank", le=10,
                             n_clients=115,
                             run_id=f"fedavg__sohrank_k1__seed{s}"))
        hetero = {"partitions": {
            "partition-mvp-k1-iid.json": {
                "between_client_variance_fraction": 0.704, "n_clients": 115},
            "partition-mvp-k1-sohrank.json": {
                "between_client_variance_fraction": 0.704, "n_clients": 115},
        }}
        out = "\n".join(sr.judge_hypotheses(runs, hetero=hetero))
        h2 = out.split("H2")[1].split("H3")[0]
        assert "处理未施加" in h2
        assert "完全相同" in h2
        assert "校准点" in h2

    def test_same_n_pair_gets_a_real_verdict(self):
        """同 N 的 sohrank/IID 配对 + 偏斜更高 → 正常按阈值判。"""
        runs = []
        for s, mae in enumerate([0.0022, 0.0024, 0.0020]):
            runs.append(_run("centralized", s, mae, n_clients=None,
                             epochs=300, run_id=f"c0__centralized__seed{s}"))
        for s, mae in enumerate([0.0020, 0.0022, 0.0018]):
            runs.append(_run("fedavg", s, mae, scheme="iid", le=10,
                             n_clients=58,
                             run_id=f"fedavg__iid_k2__seed{s}"))
        for s, mae in enumerate([0.0030, 0.0032, 0.0028]):
            runs.append(_run("fedavg", s, mae, scheme="sohrank", le=10,
                             n_clients=58,
                             run_id=f"fedavg__sohrank_k2__seed{s}"))
        hetero = {"partitions": {
            "partition-mvp-k2-iid.json": {
                "between_client_variance_fraction": 0.325, "n_clients": 58},
            "partition-mvp-k2-sohrank.json": {
                "between_client_variance_fraction": 0.572, "n_clients": 58},
        }}
        out = "\n".join(sr.judge_hypotheses(runs, hetero=hetero))
        h2 = out.split("H2")[1].split("H3")[0]
        assert "处理未施加" not in h2
        assert "57.2%" in h2 and "32.5%" in h2
        # 只有一组 sohrank、一组 IID，编号不带 N 后缀（后缀只在碰撞时出现）
        assert "P1: 偏斜 57.2%" in h2


class TestServerLrGrouping:
    """M3 ② 的 FedAdam：η 是主超参，必须进分组键。

    不放进键里的后果与 N 当年一样：η 探针的三档（实测 seed 0 上
    0.00268 / 0.00101 / 0.00340，差 3.4 倍）会被平均成一组，那张表的
    「A1」行描述的是任何一个都没跑过的配置。
    """

    @staticmethod
    def _eta_sweep_runs():
        runs = []
        for s, mae in enumerate([0.0022, 0.0024, 0.0020]):
            runs.append(_run("centralized", s, mae, n_clients=None,
                             epochs=300, run_id=f"c0__centralized__seed{s}"))
        for eta, maes in ((0.1, [0.0027, 0.0029, 0.0025]),
                          (0.01, [0.0010, 0.0012, 0.0009]),
                          (0.001, [0.0034, 0.0036, 0.0032])):
            for s, mae in enumerate(maes):
                runs.append(_run("fedadam", s, mae, scheme="iid", le=10,
                                 server_lr=eta,
                                 run_id=f"fedadam__iid__eta{eta}__seed{s}"))
        return runs

    def test_server_lr_splits_groups(self):
        groups = sr._group(self._eta_sweep_runs())
        adam = [k for k in groups if k[0] == "fedadam"]
        assert len(adam) == 3
        assert {k[6][0] for k in adam} == {0.1, 0.01, 0.001}

    def test_codes_get_eta_suffix(self):
        codes = sr._codes(list(sr._group(self._eta_sweep_runs())))
        adam = {c for k, c in codes.items() if k[0] == "fedadam"}
        assert adam == {"A1@η=0.1", "A1@η=0.01", "A1@η=0.001"}

    def test_each_row_keeps_its_own_mean(self):
        """塌成一组会把三档平均成一个数；分开后每行是自己的均值。"""
        rows = sr.build_table(self._eta_sweep_runs())
        assert len([r for r in rows if r.startswith("| A1")]) == 3
        for r in rows:
            if r.startswith("| A1"):
                assert "±" in r, "三 seeds 的行必须带标准差"

    def test_label_carries_eta(self):
        """编号只标 le 时配置列要能自解释，否则读者分不清哪行是哪个 η。"""
        key = sr._key(_run("fedadam", 0, 0.001, scheme="iid", le=10,
                           server_lr=0.01))
        assert sr._label(key) == "FedAdam (η=0.01, τ=0.001, β=(0.9,0.99)) / IID"

    def test_base_code_is_a1(self):
        assert sr._BASE_CODE[("fedadam", "iid")] == "A1"

    def test_missing_server_lr_does_not_split_fedavg(self):
        """FedAvg 没有 server_lr，键的第七维全是 None，不许把基线拆开。"""
        runs = [_run("fedavg", s, mae, scheme="iid", le=10) for s, mae in
                enumerate([0.0012, 0.0013, 0.0011])]
        groups = sr._group(runs)
        assert len(groups) == 1

    def test_tau_probe_does_not_merge_with_formal_runs(self):
        """只差 τ 的两个 run 不许塌成一组（2026-10-24 真实发生过）。

        η=0.01 的正式实验与 η=0.01、τ=1e-5 的探针只差 τ。τ 不在键里时
        两组合并，``_by_seed`` 用后一个文件静默覆盖 seed 0，配对差从
        +0.00002 变成 +0.00006——而 seeds 列还写着 8。这一条钉住
        「换掉的是数，不是行数」那种失败。
        """
        runs = [_run("centralized", s, 0.0020, n_clients=None, epochs=300,
                     run_id=f"c0__centralized__seed{s}") for s in range(3)]
        runs += [_run("fedavg", s, m, scheme="iid", le=10)
                 for s, m in enumerate([0.0013, 0.0015, 0.0011])]
        # 正式：η=0.01、τ=1e-3；探针：η=0.01、τ=1e-5（只差 τ）
        runs += [_run("fedadam", s, m, scheme="iid", le=10, server_lr=0.01)
                 for s, m in enumerate([0.0010, 0.0010, 0.0010])]
        runs += [_run("fedadam", 0, 0.0013, scheme="iid", le=10,
                      server_lr=0.01, server_tau=1e-5)]
        groups = sr._group(runs)
        adam = [k for k in groups if k[0] == "fedadam"]
        assert len(adam) == 2, "只差 τ 的两组被合并了"
        assert {k[6][3] for k in adam} == {1e-3, 1e-5}
        # 合并的另一种表现：seeds 列写成 4 而实际只有 3 个 seed
        rows = sr.build_table(runs)
        a1 = [r for r in rows if r.startswith("| A1")]
        assert len(a1) == 2
        seeds_col = {r.split("|")[6].strip() for r in a1}
        assert seeds_col == {"3", "1"}, seeds_col


class TestFedAdamVerdict:
    """judge 的 M3 ② 段：对照必须是同预算 FedAvg，不能是 C0 或别的 le。"""

    @staticmethod
    def _runs(adam_maes, avg_maes, *, adam_le=10, avg_le=10):
        runs = [_run("centralized", s, 0.0020, n_clients=None, epochs=300,
                     run_id=f"c0__centralized__seed{s}") for s in range(3)]
        runs += [_run("fedadam", s, m, scheme="iid", le=adam_le,
                      server_lr=0.01, run_id=f"fedadam__iid__seed{s}")
                 for s, m in enumerate(adam_maes)]
        runs += [_run("fedavg", s, m, scheme="iid", le=avg_le,
                      run_id=f"fedavg__iid__seed{s}")
                 for s, m in enumerate(avg_maes)]
        return runs

    def test_same_budget_control_gets_a_verdict(self):
        out = "\n".join(sr.judge_hypotheses(
            self._runs([0.0010, 0.0011, 0.0009], [0.0012, 0.0013, 0.0011])))
        assert "M3 ②" in out
        assert "vs I1 同预算" in out
        assert "改善" in out

    def test_different_le_is_not_a_control(self):
        """拿 le=3 的 FedAvg 当分母，会把训练量差异算到服务端优化器头上。"""
        out = "\n".join(sr.judge_hypotheses(
            self._runs([0.0010, 0.0011, 0.0009], [0.0120, 0.0130, 0.0110],
                       avg_le=3)))
        assert "缺同预算（local_epochs=10、N=23）的 FedAvg 对照组" in out

    def test_no_control_no_verdict(self):
        """没有 FedAvg 对照时必须明说，不许默默拿 C0 当分母。"""
        runs = [_run("centralized", s, 0.0020, n_clients=None, epochs=300,
                     run_id=f"c0__centralized__seed{s}") for s in range(3)]
        runs += [_run("fedadam", s, m, scheme="iid", le=10, server_lr=0.01)
                 for s, m in enumerate([0.0010, 0.0011, 0.0009])]
        out = "\n".join(sr.judge_hypotheses(runs))
        assert "缺同预算（local_epochs=10、N=23）的 FedAvg 对照组" in out
        assert "改善" not in out




class TestNoiseMultiplierGrouping:
    """M4 的 DP：nm 与裁剪范数必须进分组键。

    与 η 当年同一类事故：噪声扫描五档只差一个 nm，而 nm=1 与 nm=8 的
    效用预期差一个量级。漏掉这一维时被换掉的是某一行的数——`_by_seed`
    用后一个文件覆盖 seed 0，行数/seeds 列/均值全都正常。
    """

    @staticmethod
    def _sweep_runs():
        runs = [_run("centralized", s, 0.0020, n_clients=None, epochs=300,
                     run_id=f"c0__centralized__seed{s}") for s in range(3)]
        # 非 DP 的同预算基线（le=3，与 M3 ② 的等训练量对照同口径）
        runs += [_run("fedavg", s, m, scheme="iid", le=3)
                 for s, m in enumerate([0.0120, 0.0130, 0.0110])]
        # 五档：nm=0 是对照组（ε=inf），其余是 DP 臂
        for nm, maes in ((0.0, [0.0125, 0.0135, 0.0115]),
                         (1.0, [0.0130, 0.0140, 0.0120]),
                         (2.0, [0.0150, 0.0160, 0.0140]),
                         (4.0, [0.0250, 0.0260, 0.0240]),
                         (8.0, [0.0600, 0.0610, 0.0590])):
            eps = float("inf") if nm == 0 else {1.0: 152.31, 2.0: 49.11,
                                                4.0: 18.65, 8.0: 7.84}[nm]
            for s, m in enumerate(maes):
                runs.append(_run("fedavg", s, m, scheme="iid", le=3,
                                 noise_multiplier=nm, dp_epsilon=eps,
                                 run_id=f"fedavg__iid__le3__nm{nm:g}__seed{s}"))
        return runs

    def test_noise_multiplier_splits_groups(self):
        groups = sr._group(self._sweep_runs())
        dp = [k for k in groups if k[7] is not None]
        assert len(dp) == 5
        assert {k[7][0] for k in dp} == {0.0, 1.0, 2.0, 4.0, 8.0}

    def test_control_arm_does_not_merge_with_formal_runs(self):
        """nm=0 与 nm=1 只差一个数，但测的是两件事，不许塌成一组。

        合并后「DP 机制开销」会被算进「噪声开销」，而 plan §7 明确警告
        过「把梯度裁剪当成差分隐私」。
        """
        groups = sr._group(self._sweep_runs())
        dp = [k for k in groups if k[7] is not None]
        assert len({k[7][0] for k in dp}) == 5
        zero = [k for k in dp if k[7][0] == 0.0]
        assert len(zero) == 1
        assert zero[0][7][2] == float("inf")

    def test_max_grad_norm_is_part_of_the_key(self):
        """同 nm 不同裁剪范数是两个实验（敏感度上界不同，ε 也不同）。"""
        runs = [_run("centralized", s, 0.0020, n_clients=None, epochs=300,
                     run_id=f"c0__centralized__seed{s}") for s in range(3)]
        for norm in (1.0, 0.5):
            runs += [_run("fedavg", s, m, scheme="iid", le=3,
                          noise_multiplier=1.0, max_grad_norm=norm,
                          dp_epsilon=10.0 * (1.0 / norm))
                     for s, m in enumerate([0.013, 0.014, 0.012])]
        groups = sr._group(runs)
        dp = [k for k in groups if k[7] is not None]
        assert len(dp) == 2, "只差 max_grad_norm 的两组被合并了"
        assert {k[7][1] for k in dp} == {1.0, 0.5}

    def test_codes_get_nm_suffix_and_base_code_d1(self):
        codes = sr._codes(list(sr._group(self._sweep_runs())))
        dp = {c for k, c in codes.items() if k[7] is not None}
        assert dp == {f"D1@nm={nm:g}" for nm in (0, 1, 2, 4, 8)}
        # 非 DP 的同预算基线不拿 D 编号——它不是 DP 臂。夹具里只有一个 le
        # 的 FedAvg，所以 varies_le 为假、没有后缀；这不影响要断言的那件事：
        # 它的基编号仍是 I1，没有被并进 D1。
        plain = [c for k, c in codes.items()
                 if k[0] == "fedavg" and k[7] is None and k[4] == 3]
        assert plain == ["I1"], plain

    def test_label_carries_epsilon(self):
        """DOD ① 要求「每点附 ε」：ε 必须出现在配置列里。"""
        key = sr._key(_run("fedavg", 0, 0.013, scheme="iid", le=3,
                           noise_multiplier=1.0, dp_epsilon=152.31))
        assert sr._label(key) == "FedAvg / IID + DP (nm=1, C=1, ε=152.31)"
        zero = sr._key(_run("fedavg", 0, 0.0125, scheme="iid", le=3,
                            noise_multiplier=0.0, dp_epsilon=float("inf")))
        assert "无保证" in sr._label(zero)
        assert "ε=∞" not in sr._label(zero)

    def test_each_row_keeps_its_own_mean(self):
        rows = sr.build_table(self._sweep_runs())
        d1 = [r for r in rows if r.startswith("| D1")]
        assert len(d1) == 5
        # nm=8 那一行必须真的是它自己的均值（0.060），不是五档的平均
        assert any("0.06000 ±" in r for r in d1), d1


class TestDPVerdict:
    """judge 的 M4 段：单调性 + ε ≤ 8 可用性，且对照组必须是 nm=0。"""

    @staticmethod
    def _runs(maes, *, include_zero=True, include_plain=True):
        runs = [_run("centralized", s, 0.0020, n_clients=None, epochs=300,
                     run_id=f"c0__centralized__seed{s}") for s in range(3)]
        if include_plain:
            runs += [_run("fedavg", s, m, scheme="iid", le=3)
                     for s, m in enumerate([0.0120, 0.0130, 0.0110])]
        eps = {0.0: float("inf"), 1.0: 152.31, 2.0: 49.11,
               4.0: 18.65, 8.0: 7.84}
        for nm, triple in maes.items():
            if nm == 0.0 and not include_zero:
                continue
            for s, m in enumerate(triple):
                runs.append(_run("fedavg", s, m, scheme="iid", le=3,
                                 noise_multiplier=nm, dp_epsilon=eps[nm],
                                 run_id=f"fedavg__iid__le3__nm{nm:g}__seed{s}"))
        return runs

    def test_monotone_and_usable_arm_gets_a_verdict(self):
        runs = self._runs({
            0.0: [0.0125, 0.0135, 0.0115],
            1.0: [0.0130, 0.0140, 0.0120],
            8.0: [0.0140, 0.0150, 0.0130],   # ε=7.84 ≤ 8，劣化 12% ≤ 30%
        })
        out = "\n".join(sr.judge_hypotheses(runs))
        assert "M4 / H4" in out
        assert "单调性成立" in out
        assert "ε=7.84" in out
        assert "可用" in out

    def test_non_monotone_is_reported(self):
        """MAE 随 nm 下降必须报出来——那不是「DP 免费」，是某处不对。"""
        runs = self._runs({
            0.0: [0.0125, 0.0135, 0.0115],
            1.0: [0.0300, 0.0310, 0.0290],   # 反而更差
            8.0: [0.0140, 0.0150, 0.0130],   # 又变好
        })
        out = "\n".join(sr.judge_hypotheses(runs))
        assert "单调性**不成立**" in out
        assert "nm=1→8" in out

    def test_missing_control_arm_is_reported(self):
        """没有 nm=0 对照组时必须明说，不许把机制开销算到噪声头上。"""
        runs = self._runs({1.0: [0.013, 0.014, 0.012],
                           8.0: [0.060, 0.061, 0.059]}, include_zero=False)
        out = "\n".join(sr.judge_hypotheses(runs))
        assert "缺 nm=0 对照组" in out

    def test_no_arm_reaches_epsilon_target(self):
        """三档全都 ε>8 时必须直说，不拿 nm=1 冒充「ε ≤ 8 的可用点」。"""
        runs = self._runs({0.0: [0.0125, 0.0135, 0.0115],
                           1.0: [0.013, 0.014, 0.012],
                           2.0: [0.015, 0.016, 0.014]})
        out = "\n".join(sr.judge_hypotheses(runs))
        assert "ε ≤ 8 的一档不存在" in out

    def test_no_dp_runs_at_all(self):
        runs = [_run("centralized", s, 0.0020, n_clients=None, epochs=300,
                     run_id=f"c0__centralized__seed{s}") for s in range(3)]
        out = "\n".join(sr.judge_hypotheses(runs))
        assert "没有 DP 结果" in out

    def test_dp_arms_are_not_judged_under_h1(self):
        """DP 臂也在 IID 上，但不该在 H1 段再判一次。

        否则同一批 MAE 会出现在两个假设下，而读者从行上看不出它们是另一种
        处理（H1 问的是没有隐私机制时联邦能否追平）。DP 的结论只属于 M4 段，
        那里的分母是 nm=0 机制对照。
        """
        runs = self._runs({
            0.0: [0.0125, 0.0135, 0.0115],
            1.0: [0.0300, 0.0310, 0.0290],
            8.0: [0.0600, 0.0610, 0.0590],
        })
        out = "\n".join(sr.judge_hypotheses(runs))
        h1 = out.split("H1（")[1].split("H2（")[0]
        assert "nm=0" not in h1 and "nm=1" not in h1 and "nm=8" not in h1
        # 但 M4 段必须还在
        assert "M4 / H4" in out
