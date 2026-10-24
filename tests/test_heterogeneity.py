"""heterogeneity 模块测试。

这个模块的输出直接决定「H2 能不能判」。它算错的方向是**高估**异质性——
高估会让一次其实没施加处理的实验被误判成「H2 被推翻」。所以这里的测试
重点是「什么情况下必须报 0」而不是「什么情况下报一个大数」。
"""

import pytest

from flsoh.heterogeneity import (
    between_client_variance_fraction,
    heterogeneity_report,
    size_gini,
)


class TestBetweenClientVarianceFraction:

    def test_all_clients_identical_gives_zero(self):
        """每个客户端的标签分布完全一样 → 0。这是 H2 的「处理未施加」端点。"""
        labels = {c: [1.0, 1.0, 1.0] for c in ("a", "b", "c")}
        part = {"c0": ["a"], "c1": ["b"], "c2": ["c"]}
        assert between_client_variance_fraction(part, labels) == pytest.approx(0.0)

    def test_each_client_constant_and_distinct_gives_one(self):
        """每个客户端内部是常数、彼此完全不同 → 1。这是上界。"""
        labels = {"a": [0.0], "b": [1.0], "c": [2.0]}
        part = {"c0": ["a"], "c1": ["b"], "c2": ["c"]}
        assert between_client_variance_fraction(part, labels) == pytest.approx(1.0)

    def test_pooling_cells_averages_away_skew(self):
        """把两颗差异很大的电池合成一个客户端，偏斜必须下降。

        这是本仓库的核心机制：Dirichlet 划分给某个客户端 21 颗电池，
        等于把电池间的差异平均掉了，于是「Non-IID」划分反而比 IID 更均匀。
        """
        labels = {"lo": [0.0] * 10, "hi": [10.0] * 10}
        separate = {"c0": ["lo"], "c1": ["hi"]}
        pooled = {"c0": ["lo", "hi"], "c1": ["lo"]}
        # pooled 里 c1 只有 lo，c0 是平均 → 偏斜仍在但变小
        assert (between_client_variance_fraction(pooled, labels)
                < between_client_variance_fraction(separate, labels))

    def test_fully_pooled_gives_zero(self):
        """所有电池都进同一个客户端 → 0（没有客户端间差异可谈）。"""
        labels = {"a": [0.0], "b": [5.0]}
        assert between_client_variance_fraction({"c0": ["a", "b"]}, labels) == 0.0

    def test_weights_by_sample_count_not_client_count(self):
        """按样本数加权，不是按客户端数等权。

        构造：c0 有 100 个样本（均值 0.5，内部有方差），c1 只有 1 个样本
        （值 10）。等权算法会给 c1 一半的权重，但它在数据里只占 1/101。
        注意客户端内方差为 0 时这个比值恒为 1，**看不出加权**——所以这里
        必须让 c0 有内部方差。
        """
        labels = {
            "big": [0.0, 1.0] * 50,      # 100 个样本，均值 0.5，内部方差 0.25
            "small": [10.0],
        }
        part = {"c0": ["big"], "c1": ["small"]}
        frac = between_client_variance_fraction(part, labels)

        # 手算：全局均值 = (50 + 10)/101 = 60/101
        mu = 60 / 101
        # 总方差（总体口径）
        total = (
            100 * (0.25 + (0.5 - mu) ** 2) + 1 * (10 - mu) ** 2
        ) / 101
        # 加权 between 方差
        between = (100 * (0.5 - mu) ** 2 + 1 * (10 - mu) ** 2) / 101
        assert frac == pytest.approx(between / total, abs=1e-9)
        # 具体值钉住：约 0.78。等权算法会算出 >1 的数（无意义），
        # 所以这个断言同时说明「实现用的是加权口径」。
        assert frac == pytest.approx(0.781, abs=1e-3)

    def test_zero_variance_returns_zero(self):
        labels = {"a": [3.0, 3.0], "b": [3.0, 3.0]}
        part = {"c0": ["a"], "c1": ["b"]}
        assert between_client_variance_fraction(part, labels) == 0.0

    def test_unknown_cell_raises(self):
        """引用了标签表里没有的电池必须报错，不能静默当成 0。

        静默跳过会让「漏了一颗电池」看起来像「异质性偏低」，
        而这两个结论完全不同。
        """
        labels = {"a": [1.0]}
        with pytest.raises(KeyError, match="不在标签表里"):
            between_client_variance_fraction({"c0": ["a", "ghost"]}, labels)

    def test_empty_client_raises(self):
        labels = {"a": [1.0]}
        with pytest.raises(ValueError, match="没有任何标签"):
            between_client_variance_fraction({"c0": [], "c1": ["a"]}, labels)


class TestSizeGini:

    def test_equal_sizes_give_zero(self):
        assert size_gini([5, 5, 5, 5]) == pytest.approx(0.0)

    def test_maximally_unequal_is_high(self):
        """一个客户端拿走几乎所有数据 → 基尼接近 1。"""
        g = size_gini([1000] + [1] * 20)
        assert g > 0.9

    def test_known_small_case(self):
        """手算：[1, 3] → ΣΣ|x_i−x_j| = 2*|1−3| = 4；2n²x̄ = 2*4*2 = 16 → 0.25。"""
        assert size_gini([1, 3]) == pytest.approx(0.25)

    def test_empty_and_zero(self):
        assert size_gini([]) == 0.0
        assert size_gini([0, 0]) == 0.0

    def test_order_independent(self):
        assert size_gini([21, 1, 5]) == pytest.approx(size_gini([1, 5, 21]))


class TestReport:

    def test_report_contains_the_gate_fields(self):
        """判定脚本要读的字段一个都不能少。"""
        labels = {"a": [0.0], "b": [1.0]}
        part = {"c0": ["a"], "c1": ["b"]}
        r = heterogeneity_report(part, labels)
        for key in (
            "n_clients", "n_cells", "global_std", "cell_mean_std",
            "cell_within_std_median", "between_client_variance_fraction",
            "size_gini", "size_max", "size_min", "size_ratio",
        ):
            assert key in r, f"报告缺字段 {key}"

    def test_std_is_population_std(self):
        """std 用总体口径（除以 n），不是样本口径（除以 n−1）。

        这里要的是「这批样本有多散」，不是「对总体 std 的无偏估计」。
        用错口径会让上界算得偏高一点点——而这条数字就是 H2 的检验力上界。
        """
        labels = {"a": [0.0], "b": [2.0]}
        r = heterogeneity_report({"c0": ["a"], "c1": ["b"]}, labels)
        assert r["global_std"] == pytest.approx(1.0)      # 总体：sqrt(1.0)
        assert r["cell_mean_std"] == pytest.approx(1.0)

    def test_size_ratio_reports_actual_skew(self):
        """21 颗 vs 1 颗 → 21×。这是 Dirichlet α=0.5 在 N=23 时的实测形状。"""
        labels = {f"c{i}": [1.0] for i in range(22)}
        part = {"big": [f"c{i}" for i in range(21)], "small": ["c21"]}
        r = heterogeneity_report(part, labels)
        assert r["size_max"] == 21
        assert r["size_min"] == 1
        assert r["size_ratio"] == pytest.approx(21.0)
