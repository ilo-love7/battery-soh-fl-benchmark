"""图 3 的分组测试。

为什么值得单独测：这张图的结论是「精度代价随噪声上升，且在某个 ε 处
仍可用」。这个结论成立的前提是每个点确实是同一策略、同一训练量、同一
裁剪范数下的一个噪声档。与图 2 同构的教训——**分组错了而输出看起来
正常**：曲线照样画得出来，只是某一档的均值描述的是一个没跑过的配置。

只测纯函数（`_key` / `aggregate` / `epsilon_of`）：matplotlib 在
`main()` 里才导入，所以这个文件能在不装 matplotlib 的默认 CI job 里跑。
"""

from __future__ import annotations

import make_figure3 as mf


def _run(nm, mae, *, strategy="fedavg", scheme="iid", le=3, n=23,
         norm=1.0, seed=0, eps=None):
    d = {
        "strategy": strategy,
        "scheme": scheme,
        "local_epochs": le,
        "n_clients": n,
        "final_mae": mae,
        "seed": seed,
        "partition_json": f"results/partition-mvp-k5-{scheme}.json",
    }
    if nm is not None:
        # 与 scripts/run_federated.py 写盘的字段一致
        d["noise_multiplier"] = nm
        d["max_grad_norm"] = norm
        d["target_delta"] = 1e-5
        d["dp_epsilon"] = float("inf") if eps is None else eps
    return d


class TestKey:
    def test_noise_multiplier_splits_arms(self):
        """五档只差一个 nm，而 nm=1 与 nm=8 的效用预期差一个量级。"""
        keys = {mf._key(_run(nm, 0.01)) for nm in (0.0, 1.0, 2.0, 4.0, 8.0)}
        assert len(keys) == 5

    def test_control_arm_is_not_a_dp_arm(self):
        """nm=0 与 nm=1 只差一个数，但一个是无保证的对照组。

        合并后「DP 机制开销」被算进「噪声开销」，图 3 左端到非 DP 基线
        的落差会被读成「小噪声很便宜」——plan §7 明确警告过这个误读。
        """
        zero = mf._key(_run(0.0, 0.0125))
        one = mf._key(_run(1.0, 0.0130))
        assert zero != one
        assert zero[4] == 0.0 and one[4] == 1.0

    def test_no_dp_at_all_is_a_separate_group(self):
        """完全没开 DP 的 run（无 noise_multiplier 字段）必须自成一组。

        它的 nm 是 None，而 None 与 0.0 在图上是两条不同的参考线：
        一条是「机制开销」，一条是「完全没付隐私代价」。
        """
        plain = mf._key(_run(None, 0.0117))
        zero = mf._key(_run(0.0, 0.0125))
        assert plain != zero
        assert plain[4] is None and zero[4] == 0.0

    def test_max_grad_norm_splits_groups(self):
        """同 nm 不同裁剪范数是两个实验：噪声 Std = nm × C / 批大小。"""
        a = mf._key(_run(1.0, 0.013, norm=1.0, eps=152.31))
        b = mf._key(_run(1.0, 0.013, norm=0.5, eps=304.62))
        assert a != b

    def test_strategy_splits_groups(self):
        """DP 与服务端优化器正交，但两者都改过就必须分开。

        M3 ② 已经为「FedProx 混进 FedAvg 组」付过一次代价（n=8 报成 24）。
        """
        a = mf._key(_run(1.0, 0.0012, strategy="fedavg"))
        b = mf._key(_run(1.0, 0.0012, strategy="fedadam"))
        assert a != b

    def test_local_epochs_splits_budgets(self):
        """le=3 与 le=10 差一个量级，混在一起均值不代表任何一档。"""
        a = mf._key(_run(1.0, 0.013, le=3))
        b = mf._key(_run(1.0, 0.0013, le=10))
        assert a != b

    def test_n_clients_splits_groups(self):
        a = mf._key(_run(1.0, 0.013, n=23))
        b = mf._key(_run(1.0, 0.013, n=58))
        assert a != b


class TestAggregate:
    def test_same_key_accumulates_seeds(self):
        runs = [_run(1.0, 0.010 + 0.001 * s, seed=s) for s in range(8)]
        groups = mf.aggregate(runs)
        assert len(groups) == 1
        assert len(next(iter(groups.values()))) == 8

    def test_different_arms_stay_separate(self):
        runs = [_run(nm, 0.01 * nm, seed=0) for nm in (1.0, 2.0, 4.0, 8.0)]
        assert len(mf.aggregate(runs)) == 4


class TestEpsilonOf:
    def test_returns_none_for_non_dp(self):
        """非 DP 的 run 返回 None，**不是 0**。

        0 会被读成"ε=0，极强隐私保证"，而真相是"这一行没有隐私声明"。
        """
        assert mf.epsilon_of(_run(None, 0.0117)) is None

    def test_returns_inf_for_control_arm(self):
        assert mf.epsilon_of(_run(0.0, 0.0125)) == float("inf")

    def test_returns_the_recorded_value(self):
        assert mf.epsilon_of(_run(8.0, 0.06, eps=7.84)) == 7.84


class TestPickReferences:
    """参考线的选择：必须与 DP 臂同策略、同划分、同 N。

    2026-10-25 第一次出图时这里抓错了：`plain[0]` 拿到的是同 `le` 的
    **FedAdam**（0.00162）而不是 **FedAvg**（0.01165），参考线画低 7 倍，
    而图的其余部分完全正常。「同 le」只对齐训练量，策略不同就不是同一个
    实验——与图 2 的 `strategy` 漏键是同一类事故。
    """

    @staticmethod
    def _sel(**extra_plain):
        runs = []
        for nm in (0.0, 1.0, 2.0, 4.0, 8.0):
            for s in range(3):
                runs.append(_run(nm, 0.01 * (1 + nm), seed=s))
        # 非 DP 同预算的 FedAvg
        for s in range(3):
            runs.append(_run(None, 0.0117, seed=s))
        # 同 le 但不同策略/划分/N 的干扰项
        for s in range(3):
            runs.append(_run(None, 0.0016, seed=s, strategy="fedadam"))
        for s in range(3):
            runs.append(_run(None, 0.0300, seed=s, scheme="sohrank"))
        for s in range(3):
            runs.append(_run(None, 0.0500, seed=s, n=12))
        return mf.aggregate(runs)

    def test_picks_fedavg_not_fedadam(self):
        sel = self._sel()
        arms = sorted((k for k in sel if k[4] and k[4] > 0),
                      key=lambda k: k[4])
        zero, plain, err = mf.pick_references(sel, arms)
        assert err is None
        assert plain[0][0] == "fedavg"
        assert zero[0][4] == 0.0

    def test_missing_plain_baseline_is_an_error(self):
        """缺非 DP 基线必须报错，不许少画一条线就出图。"""
        runs = [_run(nm, 0.01 * (1 + nm), seed=s)
                for nm in (0.0, 1.0, 8.0) for s in range(3)]
        sel = mf.aggregate(runs)
        arms = sorted((k for k in sel if k[4] and k[4] > 0),
                      key=lambda k: k[4])
        zero, plain, err = mf.pick_references(sel, arms)
        assert err is not None and "非 DP 同预算" in err
        assert not plain

    def test_missing_zero_control_is_an_error(self):
        runs = [_run(nm, 0.01 * (1 + nm), seed=s)
                for nm in (1.0, 8.0) for s in range(3)]
        runs += [_run(None, 0.0117, seed=s) for s in range(3)]
        sel = mf.aggregate(runs)
        arms = sorted((k for k in sel if k[4] and k[4] > 0),
                      key=lambda k: k[4])
        zero, plain, err = mf.pick_references(sel, arms)
        assert err is not None and "nm=0" in err
        assert not zero
