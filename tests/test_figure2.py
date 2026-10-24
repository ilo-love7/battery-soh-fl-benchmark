"""图 2 的分组测试。

为什么值得单独测：这张图的结论是「两条线在同一 k 处的竖直距离 = 偏斜的
净效应」。这个距离成立的前提是每条线上的点确实是同一个策略、同一档 k、
同一训练量。M3 实测踩过一次：`_key` 里少了 `strategy`，FedProx 的
dirichlet run（两个 μ × 8 seeds）被并进「FedAvg / dirichlet」——图不画
那条线所以图没事，但汇总把 n=8 报成 n=24。分组错了而输出看起来正常，
正是必须用测试钉住的形态。

只测纯函数（`_key` / `aggregate`）：matplotlib 在 `main()` 里才导入，
所以这个文件能在不装 matplotlib 的默认 CI job 里跑。
"""

from __future__ import annotations

import make_figure2 as mf


def _run(strategy, scheme, k, le, mae, seed=0):
    return {
        "strategy": strategy,
        "scheme": scheme,
        "local_epochs": le,
        "final_mae": mae,
        "seed": seed,
        "partition_json": f"results/partition-mvp-k{k}-{scheme}.json",
    }


class TestKey:
    def test_strategy_splits_fedprox_from_fedavg(self):
        """FedProx 的 run 不能混进 FedAvg 的组。

        dirichlet 划分上跑过 fedprox（2 个 μ × 8 seeds）。少了 strategy
        这一维，它们会被并进「FedAvg / dirichlet」，n 从 8 变 24。
        """
        a = mf._key(_run("fedavg", "dirichlet", 5, 10, 0.001))
        b = mf._key(_run("fedprox", "dirichlet", 5, 10, 0.001))
        assert a != b

    def test_k_splits_shards(self):
        """k=1/2/5/10 四档必须分开，否则偏斜扫描没有横轴。"""
        keys = {mf._key(_run("fedavg", "sohrank", k, 10, 0.001))
                for k in (1, 2, 5, 10)}
        assert len(keys) == 4

    def test_local_epochs_splits_budgets(self):
        """le=3 与 le=10 差一个量级，混在一起均值不代表任何一档。"""
        a = mf._key(_run("fedavg", "iid", 5, 3, 0.011))
        b = mf._key(_run("fedavg", "iid", 5, 10, 0.001))
        assert a != b

    def test_dirichlet_is_not_iid(self):
        """两种划分的偏斜不同（12.2% vs 17.3%），不能归成一类。"""
        a = mf._key(_run("fedavg", "iid", 5, 10, 0.001))
        b = mf._key(_run("fedavg", "dirichlet", 5, 10, 0.001))
        assert a[1] == "iid" and b[1] == "dirichlet"

    def test_missing_partition_json_is_tolerated(self):
        """C0 之类的 run 没有 partition_json，不能因此崩。"""
        key = mf._key({"strategy": "centralized", "final_mae": 0.002})
        assert key[0] == "centralized"
        assert key[2] is None


class TestAggregate:
    def test_fedprox_does_not_join_the_fedavg_group(self):
        """端到端：16 个 fedprox run 不许把 8 个 fedavg run 的组撑大。"""
        runs = [_run("fedavg", "dirichlet", 5, 10, 0.001, seed=s)
                for s in range(8)]
        runs += [_run("fedprox", "dirichlet", 5, 10, 0.0015, seed=s)
                 for s in range(8)]
        groups = mf.aggregate(runs)
        fedavg = [v for k, v in groups.items() if k[0] == "fedavg"]
        fedprox = [v for k, v in groups.items() if k[0] == "fedprox"]
        assert len(fedavg) == 1 and len(fedavg[0]) == 8
        assert len(fedprox) == 1 and len(fedprox[0]) == 8

    def test_same_key_accumulates_seeds(self):
        runs = [_run("fedavg", "sohrank", 2, 10, 0.001 * (s + 1), seed=s)
                for s in range(3)]
        groups = mf.aggregate(runs)
        assert len(groups) == 1
        assert sorted(next(iter(groups.values()))) == [0.001, 0.002, 0.003]
