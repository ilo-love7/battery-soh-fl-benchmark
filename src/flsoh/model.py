"""模型定义：MLP + ΔQ(V) 特征向量（docs/plan.md §4.1）。

模块契约（docs/plan.md §3.2）：

=========== ==========================================
输入         输入维度（必填，无默认值）
输出         ``nn.Module``
不许做什么   **不许内置任何与数据相关的常量**
=========== ==========================================

最后一条是硬约束，所以 ``input_dim`` 是必填位置参数。1000 这个数来自
Severson et al. 2019 的 ΔQ(V) 特征构造，属于**架构选择**而不是数据常量；
换特征维度时调用方必须显式传参，编译器会逼你意识到自己改了模型。

参数量（本仓库反复引用的那个数）：:

    1000×128 + 128 + 128×64 + 64 + 64×1 + 1 = 136,449

单轮上行通信量 ``136,449 × 4 B = 545,796 B ≈ 0.52 MiB``（docs/plan.md
§4.1）。``parameter_bytes()`` 把这个算式固化成一个函数，报告里的表 2
由它算出来，不再手抄。
"""

from __future__ import annotations

import torch
from torch import nn

# 架构默认值。这些是 plan.md §4.1 的选择，不是数据常量：
# 数据相关的量（特征维度）一律由调用方显式传入。
DEFAULT_HIDDEN_DIMS: tuple[int, ...] = (128, 64)
DEFAULT_OUTPUT_DIM = 1

# 表 2 用的字节数。float32 = 4 B；换半精度训练要显式改这里，
# 不能让它悄悄跟着 dtype 漂移。
FLOAT32_BYTES = 4


def linear_param_count(in_dim: int, out_dim: int) -> int:
    """一层 ``nn.Linear(in_dim, out_dim)`` 的参数量（含 bias）。"""
    return in_dim * out_dim + out_dim


def expected_param_count(
    input_dim: int,
    hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
    output_dim: int = DEFAULT_OUTPUT_DIM,
) -> int:
    """闭式解算参数量。测试用它核对，避免"实例化一次再数"的循环依赖。"""
    dims = (input_dim, *hidden_dims, output_dim)
    return sum(linear_param_count(a, b) for a, b in zip(dims, dims[1:]))


class MLP(nn.Module):
    """1000 → 128 → 64 → 1 的 SOH 回归网络。

    - 激活用 ReLU，最后一层**不加**激活：SOH 是 [0,1] 区间的回归量，
      输出端套 sigmoid 会在接近 1 时把梯度压没，反而训不动。
    - 不用 BatchNorm/Dropout。BN 会让 FedBN 变成必选项（plan.md §5.2），
      MVP 阶段先把问题收窄到"数据异质"这一件事上。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
        output_dim: int = DEFAULT_OUTPUT_DIM,
    ) -> None:
        super().__init__()
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(f"input_dim 必须是正整数，收到 {input_dim!r}")
        if not hidden_dims:
            raise ValueError("hidden_dims 不能为空：至少留一层隐藏层")
        if any((not isinstance(d, int)) or d <= 0 for d in hidden_dims):
            raise ValueError(f"hidden_dims 必须全是正整数，收到 {hidden_dims!r}")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"output_dim 必须是正整数，收到 {output_dim!r}")

        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.output_dim = output_dim

        dims = (input_dim, *self.hidden_dims, output_dim)
        layers: list[nn.Module] = []
        for i, (a, b) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(dims) - 2:  # 最后一层不接激活
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"输入必须是 [batch, features]，收到 shape={tuple(x.shape)}")
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"输入特征维度 {x.shape[1]} 与模型 input_dim={self.input_dim} 不一致"
            )
        return self.net(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(
    input_dim: int,
    seed: int,
    hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
    output_dim: int = DEFAULT_OUTPUT_DIM,
) -> MLP:
    """带 seed 的构造入口（docs/plan.md §2.3 可复现性开关）。

    初始化也是随机性来源之一。把 seed 收敛到这一个函数里，
    训练脚本就不必各自记得 ``torch.manual_seed``——忘一次，
    "3 个 seed 的均值±标准差"就有一个点是假的。
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed 必须是非负整数，收到 {seed!r}")
    torch.manual_seed(seed)
    return MLP(input_dim, hidden_dims, output_dim)


def parameter_bytes(model: nn.Module, bytes_per_value: int = FLOAT32_BYTES) -> int:
    """模型参数的字节数。docs/plan.md 表 2 的通信量估算由它算。"""
    return sum(p.numel() for p in model.parameters()) * bytes_per_value


def round_trip_bytes(
    model: nn.Module, bytes_per_value: int = FLOAT32_BYTES
) -> int:
    """一轮通信的上下行合计：参数上行 + 广播下行 = 2 倍参数字节数。

    ``136,449 × 4 × 2 = 1,091,592 B/客户端/轮 ≈ 1.04 MiB``（plan.md §10.1）。
    """
    return 2 * parameter_bytes(model, bytes_per_value)


def wire_bytes(model: nn.Module) -> int:
    """``ArrayRecord`` 实际序列化后的字节数（docs/plan.md §10.1 表 2 的修正项）。

    实测（flwr 1.41.0，2026-10-22）：

    ============ ============ ===========
    模型          原始参数字节   ``count_bytes()``
    ============ ============ ===========
    1000→128→64→1   545,796      546,630
    1000→256→128→64 1,189,892    1,191,004
    ============ ============ ===========

    差额恒为 **139 B/张量**，与模型规模无关——那是 NumPy 序列化时要写的
    dtype 与 shape 元数据。MLP 有 6 个张量，因此多 834 B（0.15%）。

    plan.md §4.1 的 545,796 B 说的是**载荷**，本身没错；表 2 若声称"不含
    协议开销"，那这 834 B 就是已测量的协议开销。换 1D-CNN/LSTM 时张量数
    暴增，这一项要重算——那时它的占比仍然很小，但"重算"这个动作不能省。
    """
    from flwr.app import ArrayRecord

    record = ArrayRecord.from_torch_state_dict(model.state_dict())
    return int(record.count_bytes())
