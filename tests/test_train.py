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
    ServerAdam,
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


class TestServerAdam:
    """服务端 Adam（FedAdam）的测试。

    这个类要防的是两个**静默**失败，它们都不报错、曲线都照出：

    1. **忘了偏置矫正**（``eta_norm`` 里的 √(1−β2^t)/(1−β1^t)）：
       前几轮 m、v 都偏小，更新量被系统性压低，表现为「前 10 轮
       几乎不动」——而轮数正是最贵的资源。
    2. **伪梯度方向写反**（``w_global − w_avg``）：服务端每轮把参数
       推离客户端训练的方向，训练照跑，只是整体发散或停滞。

    所以下面每条测试都对着一个**手算得到的闭式解**，而不是对着
    实现自己（那等于没测）。``eta=0`` 退化为 FedAvg 那条是总对照：
       服务端更新必须是 FedAvg 与 FedAdam 的唯一差异。
    """

    def test_eta_zero_freezes_global(self):
        """η=0 时 step 必须原样返回全局参数——服务端一步都不走。

        注意这不是 FedAvg（FedAvg 是服务端 η=1 的普通梯度步），
        别把两者混为一谈。这条钉住的是「eta 真的乘上了更新量」。
        """
        g = _state(1.0)
        avg = _state(3.0)
        out = ServerAdam(eta=0.0).step(g, avg)
        assert torch.equal(out["w"], g["w"])
        assert torch.equal(out["b"], g["b"])

    def test_zero_delta_leaves_global_unchanged(self):
        """客户端没带来任何变化（avg == global）时，全局不许动。

        delta=0 ⇒ m=0、v=0 ⇒ 更新量 0/(0+τ)=0。τ>0 时精确成立；
        τ=0 时会出 NaN——那条路径由 test_rejects_nan_update 钉住。
        """
        g = _state(2.0)
        out = ServerAdam().step(g, _state(2.0))
        assert torch.allclose(out["w"], g["w"])
        assert torch.allclose(out["b"], g["b"])

    def test_first_step_with_zero_betas_is_sign_step(self):
        """β1=β2=0、τ=0 时首轮更新必须是 η·sign(delta)。

        代入闭式解：m=delta、v=delta²、eta_norm=η，
        w_new = w + η·delta/(|delta|+0) = w + η·sign(delta)。
        w 的 delta=2.0 → 1.0+0.5=1.5；b 的 delta=4.0 → 2.0+0.5=2.5。
        """
        server = ServerAdam(eta=0.5, beta_1=0.0, beta_2=0.0, tau=0.0)
        out = server.step(_state(1.0), _state(3.0))
        assert torch.allclose(out["w"], torch.full((4,), 1.5))
        assert torch.allclose(out["b"], torch.full((4,), 2.5))

    def test_constant_delta_moves_eta_per_round(self):
        """恒定伪梯度且 τ=0 时，每轮位移**恰为 η**——偏置矫正的闭式解。

        代数上：m_t = delta·(1−β1^t)，v_t = delta²·(1−β2^t)，于是
        eta_norm·m_t/√v_t = η·sign(delta)，β1^t 与 β2^t 正好约掉。

        这条同时钉住两件事：偏置矫正必须**在**（缺了它位移会随 t 增长：
        0.1、0.135、0.157……），分母也不许被悄悄改掉。期望值 1.1/1.2/1.3
        是拿计算器按公式算的，不是跑实现得来的。

        「恒定伪梯度」要求 averaged_state 跟着 global 一起走（delta 固定
        为 +2）——只给恒定的 avg 会让 global 追上来的同时 delta 缩小，
        那是另一条测试（动量的那条）才要的情形。
        """
        server = ServerAdam(eta=0.1, beta_1=0.9, beta_2=0.99, tau=0.0)
        w = _state(1.0)
        avg = _state(3.0)  # delta = +2
        for want in (1.1, 1.2, 1.3):
            w = server.step(w, avg)
            # 保持 delta = +2：avg 要跟着**新的** global 走，不是旧的
            avg = _state(float(w["w"][0]) + 2.0)
            assert abs(float(w["w"][0]) - want) < 1e-5, (server.t, float(w["w"][0]))

    def test_momentum_carries_the_previous_delta(self):
        """伪梯度变小时，β1>0 的第二轮位移必须大于 β1=0。

        这是 FedAdam 的**全部动机**：H3 实验发现客户端 Adam 动量每轮
        归零（FedAvg 只聚合参数，不聚合优化器状态），FedAdam 把动量
        放到服务端跨轮存活。若这条不过，说明动量没有跨轮累积。

        为什么用「变小的 delta」而不是恒定 delta：恒定 delta 下偏置
        矫正会让每轮位移恒为 η（见上一条），两种 β1 看不出差别；
        只有梯度变化时，动量把上一轮的方向继续带走这一点才显形。
        """
        def second_step(beta_1: float) -> float:
            server = ServerAdam(eta=0.1, beta_1=beta_1, beta_2=0.0, tau=0.0)
            w = _state(1.0)
            first = server.step(w, _state(3.0))    # delta = +2
            second = server.step(first, _state(1.5))  # delta = +0.4
            return float((second["w"] - first["w"]).abs().max())

        # β1=0：每轮只认当前 delta，位移就是 η·sign(delta)
        assert second_step(0.0) == pytest.approx(0.1, abs=1e-6)
        # β1=0.9：上一轮的大 delta 仍占 0.9 权重，位移被抬高（手算 0.2895）
        assert second_step(0.9) == pytest.approx(0.289474, abs=1e-5)

    def test_bias_correction_uses_round_number(self):
        """偏置矫正必须随轮数变化（t 从 1 开始），且等于 flwr 的闭式。

        eta_norm = η·√(1−β2^t)/(1−β1^t)。β1=0.9、β2=0.99 时它是
        **递减**的（0.1 → 0.0742 → 0.0636）：β1 衰减比分母增长更快。
        少了它、或把 t 冻结在 1，三轮都会是 0.1——这条断言逐轮值，
        不是只查单调性。
        """
        server = ServerAdam(eta=0.1, beta_1=0.9, beta_2=0.99)
        w, avg = _state(1.0), _state(3.0)
        expected = [0.1, 0.0742460, 0.0635942]
        for want in expected:
            w = server.step(w, avg)
            got = server.eta * (1 - 0.99**server.t) ** 0.5 / (1 - 0.9**server.t)
            assert abs(got - want) < 1e-6, (server.t, got)

    def test_v_starts_at_zero_first_round(self):
        """首轮 v_t 必须等于 (1−β2)·delta²，不能带着上一轮的残留。"""
        server = ServerAdam(eta=0.0, beta_2=0.99)
        server.step(_state(1.0), _state(3.0))
        # delta=2 → v = 0.01·4 = 0.04
        assert torch.allclose(server.v_t["w"], torch.full((4,), 0.04))

    def test_rejects_key_mismatch(self):
        """全局与聚合结果的键不一致必须抛错。

        静默广播会造出张冠李戴的模型：客户端的键在全局里不存在时，
        按 key 取值要么 KeyError、要么（更糟）悄悄对齐到错误的参数。
        """
        server = ServerAdam()
        g = {"w": torch.zeros(4)}
        avg = {"w": torch.zeros(4), "extra": torch.zeros(4)}
        with pytest.raises(ValueError, match="键不一致"):
            server.step(g, avg)

    def test_rejects_shape_mismatch(self):
        server = ServerAdam()
        g = {"w": torch.zeros(4)}
        avg = {"w": torch.zeros(5)}
        with pytest.raises(ValueError, match="形状不一致"):
            server.step(g, avg)

    def test_rejects_bad_hyperparameters(self):
        """超参越界必须抛错，不能静默取默认值。"""
        with pytest.raises(ValueError, match="eta"):
            ServerAdam(eta=-1.0)
        with pytest.raises(ValueError, match="beta_1"):
            ServerAdam(beta_1=1.0)
        with pytest.raises(ValueError, match="beta_2"):
            ServerAdam(beta_2=-0.1)
        with pytest.raises(ValueError, match="tau"):
            ServerAdam(tau=-1e-3)

    def test_rejects_nan_update(self):
        """τ=0 且伪梯度为 0 时必须抛错，不许把 NaN 写进结果 JSON。

        0/(√0+0) = NaN。NaN 不会让训练崩，只会静默污染之后每一轮，
        最后产出一份看起来正常的结果文件——那比崩溃贵得多。
        """
        server = ServerAdam(tau=0.0)
        with pytest.raises(ValueError, match="NaN/Inf"):
            server.step(_state(2.0), _state(2.0))

    def test_does_not_mutate_inputs(self):
        """step 不许改调用方的 state_dict——全局参数还要参与下一轮。"""
        g, avg = _state(1.0), _state(3.0)
        g_before = g["w"].clone()
        avg_before = avg["w"].clone()
        ServerAdam().step(g, avg)
        assert torch.equal(g["w"], g_before)
        assert torch.equal(avg["w"], avg_before)
