"""训练与评估的**唯一**实现（plan.md §3.3 的模块契约）。

为什么单独一个模块：这套逻辑原来写在 ``app/client_app.py`` 里，而联邦仿真
路径还要再写一遍。两份实现必然漂移，而漂移的表现是「flwr 路径和本地路径
跑出来的数不一样」——那种差异极难定位，因为两边的日志都正常。

所以这里只做一件事：把「一个模型在若干 loader 上训练/评估」写死一次，
所有调用方都走它。

## 优化器为什么默认 adam

plan 的超参是在没有真实数据时定的 ``SGD lr=1e-3``。拿到 MIT 特征表后实测
（C0 中心化基线，100 ep，同一划分同一 seed）：

============ ============
优化器        测试 MAE
============ ============
SGD 1e-3     0.0239
Adam 1e-3    0.00415
============ ============

5.8 倍。1000 维输入的尺度差异大，SGD 各向同性的更新走不动。所以默认 adam，
但保留 sgd 选项——将来换数据集时该重新量一次，别把这条当永恒结论。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from flsoh.data import label_offset
from flsoh.model import MLP

if TYPE_CHECKING:
    # 只为类型标注。运行期不导入：train.py 属于默认 CI job（只装 pytest），
    # 而 opacus 是可选依赖，顶部导入会让整个模块在没有 opacus 的机器上
    # 直接 ImportError。
    from flsoh.dp import DPTrainer


def make_optimizer(model: MLP, name: str, lr: float):
    """按名字建优化器。未知名字抛错，不静默退回默认值。"""
    name = (name or "adam").lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    raise ValueError(f"未知优化器 {name!r}（adam/sgd）")


def proximal_term(
    model: MLP, global_params: list[torch.Tensor], mu: float
) -> torch.Tensor:
    """FedProx 的近端项 ``(μ/2)·‖w − w_global‖²``。

    公开而不是私有，因为 DP 路径（``_train_epochs_dp``）也要用它——近端项
    加在损失上、与逐样本梯度怎么算无关，所以两条训练路径必须共享同一个
    实现。写成两份的话，FedProx + DP 的组合会只在其中一条上生效，而那种
    "某条路径少了近端项"的差异在曲线上看不出来。

    没有这一项时 FedProx 就退化成 FedAvg，而它退化得**完全静默**——
    训练照跑、曲线照出，只是「FedProx 组」和「FedAvg 组」的数一模一样。
    所以 ``mu > 0`` 却没给 ``global_params`` 时必须抛错。
    """
    if global_params is None:
        raise ValueError(
            "mu > 0 但没提供 global_params：FedProx 会静默退化成 FedAvg"
        )
    sq = torch.zeros((), dtype=torch.float32)
    for p, g in zip(model.parameters(), global_params):
        sq = sq + ((p - g.to(p.device)) ** 2).sum()
    return 0.5 * mu * sq


def train_epochs(
    model: MLP,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    optimizer: str = "adam",
    mu: float = 0.0,
    global_params: list[torch.Tensor] | None = None,
    dp: DPTrainer | None = None,
) -> float:
    """在 ``loader`` 上训练 ``epochs`` 轮，返回最后一个 epoch 的平均 loss。

    用 MSE 而不是 MAE：SOH 是回归量，MSE 对离群循环更敏感，而 MAE 在
    接近 0 时梯度不降。两个指标都报，但训练目标只有一个（plan.md §10.2）。

    ``dp`` 给一个 :class:`flsoh.dp.DPTrainer` 时走 DP-SGD 分支：模型、
    优化器、数据加载器全部换成 DP 包装过的那些（``model``/``loader`` 两
    个参数被忽略）。**两条分支共享同一个函数入口**是故意的——调用方
    （``run_federated``）不需要记得"DP 要调另一个函数"，而少一个分支
    就少一处静默走错路的机会。
    """
    if dp is not None:
        return _train_epochs_dp(dp, epochs=epochs, mu=mu, global_params=global_params)
    opt = make_optimizer(model, optimizer, lr)
    loss_fn = torch.nn.MSELoss()
    model.train()
    last = float("nan")
    for _ in range(epochs):
        total, seen = 0.0, 0
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x).squeeze(-1), y)
            if mu > 0:
                loss = loss + proximal_term(model, global_params, mu)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * x.shape[0]
            seen += x.shape[0]
        last = total / max(seen, 1)
    return last


def _train_epochs_dp(
    dp: DPTrainer,
    *,
    epochs: int,
    mu: float,
    global_params: list[torch.Tensor] | None,
) -> float:
    """DP-SGD 版的训练循环。与非 DP 分支逐行对应，差异只有三处：

    1. ``opt`` 是 ``DPOptimizer``：``step()`` 里做逐样本裁剪、加噪、
       再按期望批大小缩放，然后才调真正的 Adam。
    2. ``loader`` 是 Poisson 抽样的（批大小随机，期望 40 而不是 64）。
    3. 每步 ``dp.steps += 1``——这是会计器的账，漏加会让 ε 少算。

    为什么把循环放在这里而不是 ``flsoh/dp.py``：plan.md §3.3 的模块契约
    要求"一个模型在若干 loader 上训练"只有一个实现。DP 与非 DP 的差别
    全在被训练的对象上，不在循环的结构上；分成两个文件就等于允许两边
    各自演化，而演化方向一定是漂移。
    """
    loss_fn = torch.nn.MSELoss()
    dp.wrapped.train()
    last = float("nan")
    for _ in range(epochs):
        total, seen = 0.0, 0
        for x, y in dp.loader:
            dp.optimizer.zero_grad()
            loss = loss_fn(dp.wrapped(x).squeeze(-1), y)
            if mu > 0:
                loss = loss + proximal_term(dp.wrapped, global_params, mu)
            loss.backward()
            dp.optimizer.step()
            dp.steps += 1
            # Poisson 抽样下 x.shape[0] 每批都不同，所以 loss 必须按实际
            # 样本数加权，不能按批数平均——按批数平均会让"小批"和"大批"
            # 等权，而它们的信息量差一倍。
            total += float(loss.detach()) * x.shape[0]
            seen += x.shape[0]
        last = total / max(seen, 1)
    return last


def evaluate(model: MLP, loader: DataLoader) -> dict:
    """在 ``loader`` 上算 MAE / RMSE，**在 SOH 原始尺度上**。

    标签在 :class:`flsoh.data.CellDataset` 里中心化过。MAE/RMSE 平移不变，
    所以中心化不改变这两个数；显式加回偏移是为了让 ``pred`` 处于 SOH 尺度，
    将来报偏差或 R² 时口径才正确。偏移从 loader 自己读，不靠调用方传。
    """
    offset = label_offset(loader)
    model.eval()
    se = ae = n = 0.0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).squeeze(-1) + offset
            target = y + offset
            se += float(((pred - target) ** 2).sum())
            ae += float((pred - target).abs().sum())
            n += x.shape[0]
    return {
        "mae": ae / max(n, 1),
        "rmse": (se / max(n, 1)) ** 0.5,
        "n": int(n),
    }


def weighted_average(
    states: Iterable[dict[str, torch.Tensor]], weights: Iterable[float]
) -> dict[str, torch.Tensor]:
    """按权重平均若干 state_dict。这就是 FedAvg 的聚合。

    单独抽出来是为了能对它做单元测试——手写聚合最常见的错是**等权**，
    而等权在量偏斜下是错的（plan.md §14 坑 3），且不报错。

    权重为负、全零、长度不匹配都直接抛错，不做兜底。
    """
    states = list(states)
    weights = [float(w) for w in weights]
    if not states:
        raise ValueError("没有客户端参数可聚合")
    if len(states) != len(weights):
        raise ValueError(f"参数 {len(states)} 份、权重 {len(weights)} 个，不匹配")
    if any(w < 0 for w in weights):
        raise ValueError(f"权重不能为负：{weights}")
    total = sum(weights)
    if total <= 0:
        raise ValueError(f"权重之和必须为正，收到 {total}")

    out: dict[str, torch.Tensor] = {}
    for key in states[0]:
        acc = torch.zeros_like(states[0][key], dtype=torch.float32)
        for st, w in zip(states, weights):
            acc = acc + st[key].to(torch.float32) * (w / total)
        out[key] = acc
    return out


class ServerAdam:
    """服务端 Adam（FedAdam，Reddi et al. 2020，arXiv:2003.00295）。

    ## 为什么需要它

    H3 被推翻时暴露出的机理（plan.md §14 坑 2 的就地更正）：FedAvg /
    FedProx 只聚合**参数**，不聚合优化器状态，所以客户端 Adam 的动量
    每轮归零——100 轮就有 100 次「从零预热」。``local_epochs`` 因此
    不是自由超参，它决定总训练量（le=10 才追平 C0，是 C0 的 3.3 倍训练量）。

    FedAdam 把动量挪到服务端：服务端动量跨轮存活，客户端本地动量仍然
    每轮重建，但「跨轮记忆」这一项不再丢。这是对该机理的**直接检验**——
    如果动量归零是绑定约束，FedAdam 应该在更少的本地 epoch 下追平
    FedAvg；如果不是，FedAdam 不会比 FedAvg 好。

    ## 更新式

    与 flwr 1.41.0 内置 ``FedAdam`` 逐字对应（读过安装版
    ``flwr/serverapp/strategy/fedadam.py`` 与 ``fedopt.py`` 核对）：:

        delta_t  = w_avg − w_global            # 伪梯度
        m_t      = β1·m_{t−1} + (1−β1)·delta_t
        v_t      = β2·v_{t−1} + (1−β2)·delta_t²
        eta_norm = η·√(1−β2^t) / (1−β1^t)     # t 从 1 开始，偏置矫正
        w_new    = w_global + eta_norm·m_t / (√v_t + τ)

    等价性由 ``tests/test_app.py`` 里与 flwr 内置策略的逐轮对照钉住
    （与 weighted_average 同样的约定：两条路径必须逐位对得上）。

    ## 与 FedAvg 的关系

    FedAdam 与 FedAvg 的差别**只在服务端优化器**：FedAvg 是服务端
    η=1 的普通梯度步（``w + Δ``），FedAdam 是服务端 Adam。当伪梯度
    大小 |Δ| 远大于 τ 时，FedAdam 每轮位移约 ``η·sign(Δ)``——步长被
    归一化到 η，与 |Δ| 无关（恒定 Δ 时这是精确闭式解，见测试）。

    这一点正是它可能补救「客户端动量归零」的原因：有效步长不再依赖
    |Δ| 在一轮本地训练里累积多少，每轮都有约 η 的位移。

    ``η=0`` 时全局参数**冻结**（``w_new = w_global``）——注意这不是
    FedAvg，别把两者混为一谈。

    ## 三个静默失败模式

    1. **忘了偏置矫正**：前几轮 m、v 都偏小，更新量被系统性压低，
       表现为「前 10 轮几乎不动」，不报错。
    2. **delta 方向写反**（w_global − w_avg）：服务端每轮把参数推离
       客户端训练的方向，训练照跑、曲线照出，只是整体发散或停滞。
    3. **τ=0 且伪梯度为 0**：``0/(√0+0)`` 出 NaN，静默污染后面每一轮。
       所以 step 末尾检查有限性，出现 NaN/Inf 直接抛错——一个 NaN 的
       结果 JSON 比一次崩溃贵得多。
    """

    def __init__(
        self,
        eta: float = 1e-1,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-3,
    ) -> None:
        if eta < 0:
            raise ValueError(f"服务端学习率 eta 不能为负：{eta}")
        if not 0.0 <= beta_1 < 1.0:
            raise ValueError(f"beta_1 必须在 [0,1)：{beta_1}")
        if not 0.0 <= beta_2 < 1.0:
            raise ValueError(f"beta_2 必须在 [0,1)：{beta_2}")
        if tau < 0:
            raise ValueError(f"tau 不能为负：{tau}")
        self.eta = float(eta)
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.tau = float(tau)
        self.m_t: dict[str, torch.Tensor] | None = None
        self.v_t: dict[str, torch.Tensor] | None = None
        self.t = 0

    def step(
        self,
        global_state: dict[str, torch.Tensor],
        averaged_state: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """按上面的更新式推进一步，返回新的全局 state_dict。

        ``averaged_state`` 是客户端参数的加权平均（``weighted_average``
        的输出）。两个 state_dict 的键与形状必须一致——不一致说明
        客户端回了不同架构的参数，静默广播会造出张冠李戴的模型。
        """
        if set(global_state) != set(averaged_state):
            raise ValueError(
                f"全局参数与聚合结果的键不一致："
                f"{sorted(set(global_state) ^ set(averaged_state))}"
            )
        for key in global_state:
            if global_state[key].shape != averaged_state[key].shape:
                raise ValueError(
                    f"{key} 形状不一致：全局 {tuple(global_state[key].shape)}、"
                    f"聚合 {tuple(averaged_state[key].shape)}"
                )

        self.t += 1
        # 伪梯度：客户端更新方向。eta=0 时整个更新量为零，全局参数冻结。
        delta = {
            k: averaged_state[k].to(torch.float32) - global_state[k].to(torch.float32)
            for k in global_state
        }

        if self.m_t is None:
            self.m_t = {k: torch.zeros_like(v) for k, v in delta.items()}
        if self.v_t is None:
            self.v_t = {k: torch.zeros_like(v) for k, v in delta.items()}

        self.m_t = {
            k: self.beta_1 * m + (1.0 - self.beta_1) * delta[k]
            for k, m in self.m_t.items()
        }
        self.v_t = {
            k: self.beta_2 * v + (1.0 - self.beta_2) * delta[k] * delta[k]
            for k, v in self.v_t.items()
        }

        # 偏置矫正（Kingma & Ba 2014 的 α_t）：前几轮 m、v 偏向 0，
        # 不矫正会让早期更新量被系统性压低——表现是"前 10 轮不动"，
        # 而轮数本来就是最贵的资源。
        eta_norm = (
            self.eta
            * (1.0 - self.beta_2**self.t) ** 0.5
            / (1.0 - self.beta_1**self.t)
        )

        new_state = {
            k: global_state[k].to(torch.float32)
            + eta_norm * self.m_t[k] / (self.v_t[k].sqrt() + self.tau)
            for k in global_state
        }
        # τ=0 且伪梯度为 0 时上式是 0/0 = NaN。NaN 不会让训练崩，只会
        # 静默污染之后每一轮，最后产出一份"能写进表里"的垃圾结果。
        for k, v in new_state.items():
            if not bool(torch.isfinite(v).all()):
                raise ValueError(
                    f"服务端更新出现 NaN/Inf（{k}）：常见原因是 tau=0 且伪梯度为 0"
                )
        return new_state
