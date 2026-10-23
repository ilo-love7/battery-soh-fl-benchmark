"""model 模块测试。

设计意图：把 docs/plan.md §4.1 里那几个被反复引用的数字
（136,449 参数 / 545,796 B / 1,091,592 B）钉成断言。
这些数字在方案里出现多次，手抄一遍就会错一次。
"""

import pytest

# 注意顺序：importorskip 必须是本文件的第一个 import。
# 若把 numpy 之类放在它上面，CI（不装重依赖）会在 collection 阶段就
# ModuleNotFoundError，而不是干净地 skip。这个顺序被 CI 的 tests job 钉住。
torch = pytest.importorskip("torch", reason="model 需要 torch；CI 只跑纯标准库部分")

import numpy as np  # noqa: E402

from flsoh.model import (  # noqa: E402
    FLOAT32_BYTES,
    MLP,
    build_model,
    expected_param_count,
    linear_param_count,
    parameter_bytes,
    round_trip_bytes,
    wire_bytes,
)

# plan.md §4.1 的架构与数字
PLAN_INPUT_DIM = 1000
PLAN_HIDDEN = (128, 64)
PLAN_PARAMS = 136_449
PLAN_PAYLOAD_BYTES = 545_796
PLAN_ROUND_TRIP_BYTES = 1_091_592
PLAN_WIRE_BYTES = 546_630
NUMPY_METADATA_PER_TENSOR = 139
PLAN_TENSOR_COUNT = 6


class TestParamCount:
    def test_plan_architecture_parameter_count(self):
        assert expected_param_count(PLAN_INPUT_DIM, PLAN_HIDDEN) == PLAN_PARAMS

    def test_instantiated_model_matches_closed_form(self):
        assert MLP(PLAN_INPUT_DIM, PLAN_HIDDEN).num_parameters() == PLAN_PARAMS

    def test_layer_breakdown(self):
        # 1000×128+128 + 128×64+64 + 64×1+1
        assert (
            linear_param_count(1000, 128)
            + linear_param_count(128, 64)
            + linear_param_count(64, 1)
        ) == PLAN_PARAMS

    def test_scales_with_input_dim(self):
        assert expected_param_count(100, PLAN_HIDDEN) < PLAN_PARAMS


class TestConstruction:
    def test_input_dim_is_required(self):
        # 模块契约：不许内置数据相关常量。input_dim 无默认值，
        # 调用方必须显式给，否则 TypeError。
        with pytest.raises(TypeError):
            MLP()  # type: ignore[call-arg]

    def test_rejects_non_positive_input_dim(self):
        for bad in (0, -1, 2.5, "1000", None):
            with pytest.raises((ValueError, TypeError)):
                MLP(bad, PLAN_HIDDEN)

    def test_rejects_empty_hidden_dims(self):
        with pytest.raises(ValueError):
            MLP(PLAN_INPUT_DIM, ())

    def test_rejects_bad_output_dim(self):
        with pytest.raises(ValueError):
            MLP(PLAN_INPUT_DIM, PLAN_HIDDEN, 0)

    def test_no_batchnorm_or_dropout(self):
        # FedBN 不是内置策略（plan.md §5.2）。MVP 的 MLP 若含 BN 层，
        # "数据异质"这一个变量就被模型结构污染了。
        m = MLP(PLAN_INPUT_DIM, PLAN_HIDDEN)
        kinds = {type(mod).__name__ for mod in m.modules()}
        assert "BatchNorm1d" not in kinds
        assert "Dropout" not in kinds

    def test_final_layer_has_no_activation(self):
        # SOH 是 [0,1] 回归量；输出端套 sigmoid 会在接近 1 时压没梯度。
        m = MLP(PLAN_INPUT_DIM, PLAN_HIDDEN)
        assert not isinstance(m.net[-1], torch.nn.ReLU)
        assert isinstance(m.net[-1], torch.nn.Linear)


class TestForward:
    def test_output_shape(self):
        m = MLP(PLAN_INPUT_DIM, PLAN_HIDDEN)
        x = torch.zeros(4, PLAN_INPUT_DIM)
        assert tuple(m(x).shape) == (4, 1)

    def test_can_output_negative(self):
        # 无输出激活的证据：同样的输入可能给出负值
        m = MLP(8, (16,))
        with torch.no_grad():
            outs = [float(m(torch.randn(1, 8))) for _ in range(50)]
        assert any(o < 0 for o in outs)

    def test_rejects_wrong_feature_dim(self):
        m = MLP(PLAN_INPUT_DIM, PLAN_HIDDEN)
        with pytest.raises(ValueError):
            m(torch.zeros(2, 999))

    def test_rejects_wrong_rank(self):
        m = MLP(8, (16,))
        with pytest.raises(ValueError):
            m(torch.zeros(8))


class TestBuildModel:
    def test_same_seed_same_weights(self):
        a = build_model(64, seed=7)
        b = build_model(64, seed=7)
        for pa, pb in zip(a.parameters(), b.parameters()):
            assert torch.equal(pa, pb)

    def test_different_seed_different_weights(self):
        a = build_model(64, seed=7)
        b = build_model(64, seed=8)
        assert any(not torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters()))

    def test_rejects_negative_seed(self):
        with pytest.raises(ValueError):
            build_model(64, seed=-1)


class TestCommunicationBudget:
    """plan.md §4.1 与 §10.1 表 2 的数字必须由代码算出来。"""

    def test_payload_bytes(self):
        assert parameter_bytes(MLP(PLAN_INPUT_DIM, PLAN_HIDDEN)) == PLAN_PAYLOAD_BYTES

    def test_round_trip_bytes(self):
        m = MLP(PLAN_INPUT_DIM, PLAN_HIDDEN)
        assert round_trip_bytes(m) == PLAN_ROUND_TRIP_BYTES
        assert round_trip_bytes(m) == 2 * PLAN_PAYLOAD_BYTES

    def test_wire_bytes_matches_flwr_measurement(self):
        # 2026-10-22 实测：flwr 1.41.0 的 ArrayRecord.count_bytes()
        assert wire_bytes(MLP(PLAN_INPUT_DIM, PLAN_HIDDEN)) == PLAN_WIRE_BYTES

    def test_wire_overhead_is_numpy_metadata_per_tensor(self):
        m = MLP(PLAN_INPUT_DIM, PLAN_HIDDEN)
        delta = wire_bytes(m) - parameter_bytes(m)
        assert delta == NUMPY_METADATA_PER_TENSOR * PLAN_TENSOR_COUNT

    def test_wire_overhead_scales_with_tensor_count_not_size(self):
        small = MLP(100, PLAN_HIDDEN)      # 6 张量
        deep = MLP(1000, (256, 128, 64))   # 8 张量
        d_small = wire_bytes(small) - parameter_bytes(small)
        d_deep = wire_bytes(deep) - parameter_bytes(deep)
        assert d_small == 6 * NUMPY_METADATA_PER_TENSOR
        assert d_deep == 8 * NUMPY_METADATA_PER_TENSOR

    def test_four_bytes_per_value(self):
        assert FLOAT32_BYTES == 4

    def test_total_volume_formula(self):
        """plan.md §10.1：N=23、100 轮 → 2,510,661,600 B ≈ 2.34 GiB。

        这里钉住一个已经犯过的错误：v1 方案写「N=25、100 轮 → 约 2.60 GiB」，
        而 N=25 本身就是 124÷5 的取整。即使用 N=25 算，结果也是 2.54 GiB
        而不是 2.60。派生表抄了错的输入，就会一直错下去。

        v2.1 又改了一次输入：实测 140 颗、去 5 颗跨批次延续得 135 颗、留出 20
        颗测试 → (135-20)/5 = 23 个客户端。N 从 20 变 23，总量从 2.03 变 2.34。
        同一个教训重复了一次：**N 不是常数，它是颗数和 k 的函数**。
        """
        m = MLP(PLAN_INPUT_DIM, PLAN_HIDDEN)
        n_clients, num_rounds = 23, 100
        total = round_trip_bytes(m) * n_clients * num_rounds
        assert total == 2_510_661_600
        assert total / 1024**3 == pytest.approx(2.34, abs=0.01)

    def test_old_n25_figure_was_wrong(self):
        """回归：N=25 时的正确值是 2.54 GiB，不是方案里写的 2.60。"""
        m = MLP(PLAN_INPUT_DIM, PLAN_HIDDEN)
        total = round_trip_bytes(m) * 25 * 100
        assert total / 1024**3 == pytest.approx(2.54, abs=0.01)
        assert total / 1024**3 != pytest.approx(2.60, abs=0.01)
