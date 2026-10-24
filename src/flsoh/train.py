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

import torch
from torch.utils.data import DataLoader

from flsoh.data import label_offset
from flsoh.model import MLP


def make_optimizer(model: MLP, name: str, lr: float):
    """按名字建优化器。未知名字抛错，不静默退回默认值。"""
    name = (name or "adam").lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    raise ValueError(f"未知优化器 {name!r}（adam/sgd）")


def _proximal_term(
    model: MLP, global_params: list[torch.Tensor], mu: float
) -> torch.Tensor:
    """FedProx 的近端项 ``(μ/2)·‖w − w_global‖²``。

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
) -> float:
    """在 ``loader`` 上训练 ``epochs`` 轮，返回最后一个 epoch 的平均 loss。

    用 MSE 而不是 MAE：SOH 是回归量，MSE 对离群循环更敏感，而 MAE 在
    接近 0 时梯度不降。两个指标都报，但训练目标只有一个（plan.md §10.2）。
    """
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
                loss = loss + _proximal_term(model, global_params, mu)
            loss.backward()
            opt.step()
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
