"""MIT 数据加载 + Dataset/DataLoader（docs/plan.md §3.2 模块契约）。

=========== ==========================================================
输入         电池 ID、本地训练/验证电池集合
输出         该客户端的 ``DataLoader`` 对
不许做什么   **不许做全局标准化**（docs/plan.md §14 坑 1）
=========== ==========================================================

纯特征数学（列名、``delta_q_features``、``make_v_grid``、``validate_feature_table``、
``LocalStandardizer``、``FixedScaler``）在 :mod:`flsoh.features`，本模块从那里
re-export——它们不依赖 torch，而 ``build_features.py`` 要在没装 torch 的 CI job
里跑。

## 为什么"不许全局标准化"是一条硬规则

特征的均值/方差若用全量数据算，等于把整个数据集的统计量上传了一遍——
这本身就是信息泄露，而且它绕过了联邦学习想防的全部东西。

所以本模块把标准化做成 ``LocalStandardizer``：它**只拟合某个客户端
自己的训练电池**，并且把 ``fitted_on`` 记下来。``make_client_loaders``
在拟合后断言 val 电池不在 ``fitted_on`` 里——不是靠注释提醒，是靠断言。

由此引入的输入分布差异（各客户端 scale 不一致）不是 bug，**它就是
Non-IID 的一部分**，要写进报告（plan.md §14 坑 1）。

## 全局测试集怎么标准化

服务器端 ``evaluate_fn`` 要用统一测试集评估全局模型，但任何客户端的
统计量都不能用。三种选择，本模块都支持，并在 docstring 里说清代价：

``"test"``
    用测试集自己的统计量。评估时缩放，不参与训练。代价：测试集分布
    信息进入了评估流程，指标会略微乐观。
``"fixed"``
    调用方给一个与数据无关的固定 scale（如标称容量、固定电压区间宽度）。
    **推荐**——真正零泄露，且与客户端无关。
``"none"``
    不缩放。只有特征本身量纲一致时可用。

默认 ``"test"``，因为这是最常见的做法且代价可量化；但配置项显式存在，
让人必须选一次。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from flsoh.features import (  # noqa: F401 —— re-export，调用方一直从这里 import
    CELL_COL,
    CYCLE_COL,
    FEATURE_PREFIX,
    SOH_CENTER,
    SOH_COL,
    DataContractError,
    FeatureTable,
    FixedScaler,
    LocalStandardizer,
    delta_q_curve,
    delta_q_features,
    feature_columns,
    load_feature_table,
    make_v_grid,
    validate_feature_table,
)
from flsoh.partition import make_cell_ids


# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------


class CellDataset(Dataset):
    """按电池 ID 过滤的 (features, soh) 数据集。

    只认 ``cell_id`` 集合，不做任何时间切分——时间切分的规则是
    plan.md §6.2 规则 1（以电池为最小原子，客户端内再按时间切 train/val），
    属于调用方的策略，不是加载器该决定的事。

    **标签在这里中心化**（``y = soh - label_center``，默认 :data:`SOH_CENTER`）。
    放在这一层的理由：它是数据契约的一部分，所有训练与评估路径都必须经过
    同一个变换；只要有一处忘了加回，指标就会莫名其妙地差 1.0。
    ``label_center`` 是实例属性，评估方从 loader.dataset 读它加回，
    不靠调用方自己记。

    不中心化的代价是实测过的：C0 中心化基线 300 ep 的测试 MAE 从 0.00155
    劣化到 0.11333（73 倍），因为损失几乎全被「输出偏置差 1.0」占据。
    """

    def __init__(
        self,
        table: FeatureTable,
        cell_ids,
        scaler=None,
        *,
        label_center: float = SOH_CENTER,
        cycles=None,
    ) -> None:
        wanted = list(dict.fromkeys(cell_ids))
        unknown = set(wanted) - set(table.cells)
        if unknown:
            raise DataContractError(f"电池不在特征表里：{sorted(unknown)[:5]}")
        if not isinstance(label_center, (int, float)):
            raise DataContractError(f"label_center 必须是数值，收到 {label_center!r}")
        sub = table.subset(wanted)
        if cycles is not None:
            # 客户端内按时间切 train/val（plan.md §6.2 规则 1）。切的是**循环号**
            # 而不是电池：同一颗电池的前 80% 循环进训练、后 20% 进验证，
            # 这样验证集测的是"能不能外推到未来循环"，而不是"能不能认出另一颗电池"。
            keep = np.isin(sub[CYCLE_COL].to_numpy(), np.asarray(list(cycles)))
            sub = sub[keep]
            if len(sub) == 0:
                raise DataContractError(
                    f"按循环过滤后一行不剩：cells={wanted[:3]} cycles={sorted(cycles)[:5]}…"
                )
        self.cell_ids = wanted
        self.cycles = sub[CYCLE_COL].to_numpy()
        self.label_center = float(label_center)
        self.y = torch.tensor(
            sub[SOH_COL].to_numpy(dtype=float) - self.label_center,
            dtype=torch.float32,
        )
        x = sub.loc[:, list(table.feats)].to_numpy(dtype=float)
        if scaler is not None:
            x = scaler.transform(x)
        self.x = torch.tensor(x, dtype=torch.float32)
        self.input_dim = x.shape[1]

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def label_offset(loader: DataLoader) -> float:
    """读回 ``loader`` 用的标签中心化偏移，评估时必须加回。

    从 ``loader.dataset`` 读而不是让调用方自己传一个常量，是因为
    「训练时减了什么」和「评估时加什么」必须是同一个数。调用方各自记一份，
    迟早会漂移，而漂移的表现是指标整体差 1.0——看起来像模型崩了。

    读不到就抛错，不返回 0：返回 0 会让指标静默地错 1.0，比崩掉更难查。
    """
    dataset = getattr(loader, "dataset", None)
    center = getattr(dataset, "label_center", None)
    if center is None:
        raise DataContractError(
            "loader 的 dataset 没有 label_center 属性，评估无法还原 SOH 尺度。"
            "请用 CellDataset 构造数据集，不要自己拼 TensorDataset。"
        )
    return float(center)


@dataclass
class ClientData:
    """一个客户端的训练/验证 DataLoader 对，附带审计信息。"""
    train_loader: DataLoader
    val_loader: DataLoader
    train_cells: tuple[str, ...]
    val_cells: tuple[str, ...]
    standardizer_fitted_on: frozenset[str]
    n_train: int
    n_val: int


def make_client_loaders(
    table: FeatureTable,
    train_cell_ids,
    val_cell_ids,
    *,
    batch_size: int = 64,
    seed: int = 0,
    num_workers: int = 0,
    label_center: float = SOH_CENTER,
) -> ClientData:
    """构造一个客户端的 DataLoader 对（plan.md §3.2 的模块契约入口）。

    标准化器只拟合 ``train_cell_ids``，随后断言 ``val_cell_ids`` 没参与。
    ``shuffle`` 用派生自 ``seed`` 的 generator，不用全局随机——否则
    plan.md §2.3 的"固定 seed"是一条空话。

    标签按 ``label_center`` 中心化（默认 :data:`SOH_CENTER`），见
    :class:`CellDataset`。
    """
    if batch_size <= 0:
        raise DataContractError(f"batch_size 必须为正，收到 {batch_size}")
    train_cells = list(dict.fromkeys(train_cell_ids))
    val_cells = list(dict.fromkeys(val_cell_ids))
    overlap = set(train_cells) & set(val_cells)
    if overlap:
        raise DataContractError(
            f"训练与验证电池有交集：{sorted(overlap)[:5]}。"
            "同一颗电池不能同时做训练和验证，否则 val 指标虚高"
        )

    scaler = LocalStandardizer().fit(table, train_cells)
    scaler.check_no_leak(val_cells, label="本地验证集")

    train_ds = CellDataset(table, train_cells, scaler, label_center=label_center)
    val_ds = (
        CellDataset(table, val_cells, scaler, label_center=label_center)
        if val_cells
        else None
    )

    g = torch.Generator()
    g.manual_seed(int(seed))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=g, num_workers=num_workers
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if val_ds is not None
        else None
    )
    return ClientData(
        train_loader=train_loader,
        val_loader=val_loader,
        train_cells=tuple(train_cells),
        val_cells=tuple(val_cells),
        standardizer_fitted_on=scaler.fitted_on,
        n_train=len(train_ds),
        n_val=0 if val_ds is None else len(val_ds),
    )


def split_cycles_by_time(
    table: FeatureTable, cell_ids, *, train_fraction: float = 0.8
) -> tuple[list[int], list[int]]:
    """把该客户端所有电池的循环号按时间切成 train / val 两段。

    plan.md §6.2 规则 1 要求「客户端内再按时间切 train/val」。按**电池**切
    （``cells[:split]`` / ``cells[split:]``）看起来也像切分，但它测的是
    「能不能认出另一颗电池」，而 SOH 预测真正要问的是「能不能外推到未来
    循环」。两者不是一回事，且前者会给出虚高的本地 val 指标。

    切的是循环号而不是行：同一颗电池的前 80% 循环进训练、后 20% 进验证，
    所以 train 与 val 的电池集合**相同**——这正是按时间切与按电池切的
    关键区别，调用方不要拿「有没有交集」来判泄漏（那种检查在这里是反的）。
    """
    if not 0.0 < train_fraction < 1.0:
        raise DataContractError(f"train_fraction 必须在 (0,1)，收到 {train_fraction}")
    wanted = list(dict.fromkeys(cell_ids))
    if not wanted:
        raise DataContractError("cell_ids 为空，无法按时间切分")
    cycles = np.sort(np.unique(table.subset(wanted)[CYCLE_COL].to_numpy()))
    if len(cycles) < 2:
        raise DataContractError(
            f"只有 {len(cycles)} 个循环号，无法切成 train/val"
        )
    cut = int(len(cycles) * train_fraction)
    cut = min(max(cut, 1), len(cycles) - 1)  # 两边都至少留一个循环
    return cycles[:cut].tolist(), cycles[cut:].tolist()


def make_client_loaders_by_time(
    table: FeatureTable,
    cell_ids,
    *,
    batch_size: int = 64,
    seed: int = 0,
    num_workers: int = 0,
    label_center: float = SOH_CENTER,
    train_fraction: float = 0.8,
    scaler=None,
) -> ClientData:
    """按时间切 train/val 的客户端 loader 对（plan.md §6.2 规则 1）。

    与 :func:`make_client_loaders` 的区别只有一处：这里的 train/val 是
    **同一批电池的不同循环**，不是两批电池。所以不套用「train 与 val 电池
    不得有交集」那条断言——那条断言属于按电池切的场景。

    ``scaler`` 显式给出时直接用它，不再本地拟合。**这不是给生产用的**：
    传一个从全部客户端数据拟合的 scaler 就等于全局标准化（plan.md §14 坑 1）。
    它存在的唯一理由是跑对照实验，量化「本地标准化本身带来多少劣化」——
    那个数不给出来，就分不清联邦掉点是数据异质还是标准化噪声。
    """
    if batch_size <= 0:
        raise DataContractError(f"batch_size 必须为正，收到 {batch_size}")
    cells = list(dict.fromkeys(cell_ids))
    if not cells:
        raise DataContractError("cell_ids 为空，无法构造客户端数据")
    train_cycles, val_cycles = split_cycles_by_time(
        table, cells, train_fraction=train_fraction
    )

    if scaler is None:
        scaler = LocalStandardizer().fit(table, cells)

    train_ds = CellDataset(
        table, cells, scaler, label_center=label_center, cycles=train_cycles
    )
    val_ds = CellDataset(
        table, cells, scaler, label_center=label_center, cycles=val_cycles
    )

    g = torch.Generator()
    g.manual_seed(int(seed))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return ClientData(
        train_loader=train_loader,
        val_loader=val_loader,
        train_cells=tuple(cells),
        val_cells=tuple(cells),
        standardizer_fitted_on=scaler.fitted_on,
        n_train=len(train_ds),
        n_val=len(val_ds),
    )


def make_global_test_loader(
    table: FeatureTable,
    test_cell_ids,
    *,
    strategy: str = "test",
    fixed_scale: np.ndarray | None = None,
    batch_size: int = 64,
    label_center: float = SOH_CENTER,
) -> tuple[DataLoader, str]:
    """服务器端全局测试集的 DataLoader（plan.md §11.2 的 C0/C1 共用）。

    返回 ``(loader, 实际使用的策略名)``。策略三选一，见模块 docstring；
    这里刻意返回实际用的策略，让调用方把它写进结果 JSON——
    评估口径变了而没记录，是比指标低更严重的问题。
    """
    cells = list(dict.fromkeys(test_cell_ids))
    unknown = set(cells) - set(table.cells)
    if unknown:
        raise DataContractError(f"测试电池不在特征表里：{sorted(unknown)[:5]}")

    if strategy == "none":
        scaler = None
    elif strategy == "fixed":
        if fixed_scale is None:
            raise DataContractError("strategy='fixed' 必须提供 fixed_scale")
        scaler = FixedScaler(fixed_scale)
    elif strategy == "test":
        scaler = LocalStandardizer().fit(table, cells)
    else:
        raise DataContractError(f"未知的全局测试标准化策略 {strategy!r}（test/fixed/none）")

    ds = CellDataset(table, cells, scaler, label_center=label_center)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader, strategy


# ---------------------------------------------------------------------------
# 合成数据（测试用，不参与任何实验结论）
# ---------------------------------------------------------------------------


def synthetic_feature_table(
    n_cells: int = 20,
    cycles_per_cell: int = 40,
    input_dim: int = 16,
    seed: int = 0,
    capacity_fade: float = 0.002,
) -> FeatureTable:
    """生成一个形状正确、含容量衰减趋势的合成特征表。

    **只用于测试加载器与划分逻辑，不用于任何实验结论。** 它存在的理由是：
    没有它，``data.py`` 的每一条规则（不全局标准化、train/val 不交集、
    按电池过滤）都要等真实 MIT 数据到位才能验证。

    电池 ID 由 :func:`flsoh.partition.make_cell_ids` 生成——**不要在这里
    另写一套格式**。合成表用 ``cell-0000`` 而划分用 ``cell-000`` 时，
    两者对不上账，``check_partition.py`` 会报一片 P-UNKNOWN，而那是格式
    不一致而不是真的数据错误。这类错误极难排查，所以格式只允许有一个来源。

    衰减项让 SOH 与特征相关，否则标准化测试会退化成"除以一个常数"，
    抓不到真正的错误。
    """
    if n_cells <= 0 or cycles_per_cell <= 1 or input_dim <= 0:
        raise DataContractError("n_cells>0, cycles_per_cell>1, input_dim>0")
    rng = np.random.default_rng(seed)
    rows = []
    for cell_id in make_cell_ids(n_cells):
        cell_offset = rng.normal(0.0, 0.5, size=input_dim)
        for k in range(cycles_per_cell):
            soh = max(0.0, 1.0 - capacity_fade * k + rng.normal(0.0, 0.01))
            feats = cell_offset + soh * rng.normal(0.0, 1.0, size=input_dim)
            row = {CELL_COL: cell_id, CYCLE_COL: k, SOH_COL: float(soh)}
            row.update({f"{FEATURE_PREFIX}{i}": float(feats[i]) for i in range(input_dim)})
            rows.append(row)
    return FeatureTable.from_frame(pd.DataFrame(rows))
