"""训练与聚合的测试（`flsoh/train.py`）。

这个文件存在的原因很具体：**手写聚合是 FL 复现里最高频的错误来源**。
plan.md §14 坑 3 写的就是它——等权平均不报错、曲线正常、指标看着也合理，
只是所有结论都错。flwr 内置策略用 `weighted_by_key="num-examples"` 规避，
而本地联邦循环（`scripts/run_federated.py`）是手写的，所以必须单独钉住。

三类断言：

1. **加权聚合与手算逐位相等**，且等权与加权**不同**（否则测了等于没测）；
2. **非法输入抛错**，不静默兜底（权重为负、全零、长度不匹配）；
3. **FedProx 的近端项真的生效**，且 μ=0 时与 FedAvg 完全一致——
   这条是防"FedProx 静默退化成 FedAvg"，那种情况下列出来的 FedProx
   数字会和 FedAvg 一模一样，而没有任何报错。
"""

import pytest

# importorskip 必须是第一个 import（CI 不装 torch 时要干净 skip，不是 collection 崩）
torch = pytest.importorskip("torch", reason="train 需要 torch")

from flsoh.model import build_model  # noqa: E402
from flsoh.train import (  # noqa: E402
    make_optimizer,
    train_epochs,
    weighted_average,
)


def _state(value: float, dim: int = 4) -> dict:
    return {
        "w": torch.full((dim,), value, dtype=torch.float32),
        "b": torch.full((dim,), value * 2, dtype=torch.float32),
    }


class TestWeightedAverage:
    def test_matches_hand_computation(self):
        """加权平均必须与手算逐位相等。

        两个客户端、权重 1:3 → 期望 0.25·A + 0.75·B。
        这个数不能靠"看起来对"来判断，所以写死期望值。
        """
        a, b = _state(1.0), _state(3.0)
        out = weighted_average([a, b], [1.0, 3.0])
        assert torch.allclose(out["w"], torch.full((4,), 2.5))
        assert torch.allclose(out["b"], torch.full((4,), 5.0))

    def test_weighted_differs_from_unweighted(self):
        """反面对照：加权与等权必须给出不同结果。

        如果这条过了但上一条也过，说明权重真的被用上了。若两条都"过"
        而实现其实是等权，上一条会红——所以两条一起才能锁住。
        """
        a, b = _state(1.0), _state(3.0)
        weighted = weighted_average([a, b], [1.0, 3.0])
        equal = weighted_average([a, b], [1.0, 1.0])
        assert not torch.allclose(weighted["w"], equal["w"])
        assert torch.allclose(equal["w"], torch.full((4,), 2.0))

    def test_single_client_is_identity(self):
        a = _state(7.0)
        out = weighted_average([a], [123.0])
        assert torch.allclose(out["w"], a["w"])

    def test_weights_need_not_sum_to_one(self):
        """权重只按比例用，不必先归一化。"""
        a, b = _state(1.0), _state(3.0)
        out = weighted_average([a, b], [10.0, 30.0])
        assert torch.allclose(out["w"], torch.full((4,), 2.5))

    def test_rejects_negative_weight(self):
        with pytest.raises(ValueError, match="负"):
            weighted_average([_state(1.0), _state(2.0)], [1.0, -1.0])

    def test_rejects_zero_total_weight(self):
        with pytest.raises(ValueError, match="权重之和"):
            weighted_average([_state(1.0), _state(2.0)], [0.0, 0.0])

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="不匹配"):
            weighted_average([_state(1.0), _state(2.0)], [1.0])

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="没有客户端"):
            weighted_average([], [])

    def test_does_not_mutate_inputs(self):
        a = _state(1.0)
        before = a["w"].clone()
        weighted_average([a, _state(3.0)], [1.0, 1.0])
        assert torch.equal(a["w"], before)


class TestOptimizerChoice:
    def test_unknown_name_raises(self):
        """未知优化器必须抛错，不能静默退回默认值。

        静默退回会让"我跑的是 adam"变成一句无法验证的话。
        """
        model = build_model(8, seed=0)
        with pytest.raises(ValueError, match="未知优化器"):
            make_optimizer(model, "lamb", 1e-3)

    def test_adam_and_sgd_are_different_types(self):
        model = build_model(8, seed=0)
        assert isinstance(make_optimizer(model, "adam", 1e-3), torch.optim.Adam)
        assert isinstance(make_optimizer(model, "sgd", 1e-3), torch.optim.SGD)


def _toy_loader(n: int = 32, dim: int = 8):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, dim, generator=g)
    y = x[:, 0] * 0.5 + 0.1
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=8
    )


class TestFedProx:
    def test_mu_zero_matches_plain_training(self):
        """μ=0 时 FedProx 路径必须与普通训练**完全相同**。

        同一个初始模型、同一个 loader、同一个 seed，两次训练后的参数
        必须逐位相等——否则说明近端项在 μ=0 时也动了手。
        """
        loader = _toy_loader()
        m1 = build_model(8, seed=0)
        m2 = build_model(8, seed=0)
        train_epochs(m1, loader, epochs=1, lr=1e-3, optimizer="adam", mu=0.0)
        train_epochs(m2, loader, epochs=1, lr=1e-3, optimizer="adam")
        for p, q in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p, q)

    def test_mu_positive_changes_update(self):
        """μ>0 时参数必须与 μ=0 不同——否则近端项没生效。

        这是「FedProx 静默退化成 FedAvg」的唯一可观测证据。若这条红，
        意味着报告里的 FedProx 组其实是 FedAvg 组。
        """
        loader = _toy_loader()
        m0 = build_model(8, seed=0)
        m1 = build_model(8, seed=0)
        train_epochs(m0, loader, epochs=1, lr=1e-3, optimizer="adam", mu=0.0)

        ref = [p.detach().clone() for p in m1.parameters()]
        train_epochs(
            m1, loader, epochs=1, lr=1e-3, optimizer="adam", mu=0.5, global_params=ref
        )
        assert any(
            not torch.equal(p, q) for p, q in zip(m0.parameters(), m1.parameters())
        )

    def test_mu_positive_without_global_params_raises(self):
        """μ>0 却没给 global_params 必须抛错。

        否则近端项会被跳过，而 FedProx 悄悄变成 FedAvg。
        """
        model = build_model(8, seed=0)
        with pytest.raises(ValueError, match="global_params"):
            train_epochs(
                model, _toy_loader(), epochs=1, lr=1e-3, mu=0.1, global_params=None
            )

    def test_larger_mu_stays_closer_to_global(self):
        """μ 越大，训完离全局参数越近——这是近端项的定义。

        不测绝对距离（那依赖数据），只测单调性：μ=1.0 的距离必须小于 μ=0.1。
        """
        def dist_after(mu: float) -> float:
            model = build_model(8, seed=0)
            ref = [p.detach().clone() for p in model.parameters()]
            train_epochs(
                model,
                _toy_loader(),
                epochs=3,
                lr=1e-2,
                optimizer="sgd",
                mu=mu,
                global_params=ref,
            )
            return sum(
                float(((p.detach() - g) ** 2).sum())
                for p, g in zip(model.parameters(), ref)
            )

        assert dist_after(1.0) < dist_after(0.1)
