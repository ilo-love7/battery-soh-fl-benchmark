"""MIT 数据加载 + ΔQ(V) 特征提取（docs/plan.md §3.2 模块契约）。

=========== ==========================================================
输入         电池 ID、本地训练/验证电池集合
输出         该客户端的 ``DataLoader`` 对
不许做什么   **不许做全局标准化**（docs/plan.md §14 坑 1）
=========== ==========================================================

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

from flsoh.partition import make_cell_ids

# 特征表列名。f{i} 的个数由文件决定，不写死——输入维度是模型的事，
# 不是数据加载器的事（plan.md §3.2 对 model.py 的同款约束）。
CELL_COL = "cell_id"
CYCLE_COL = "cycle"
SOH_COL = "soh"
FEATURE_PREFIX = "f"


class DataContractError(ValueError):
    """特征表不符合约定（缺列、空、维度不一致）。"""


# ---------------------------------------------------------------------------
# 特征表
# ---------------------------------------------------------------------------


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX) and c[1:].isdigit()]
    return sorted(cols, key=lambda c: int(c[1:]))


def validate_feature_table(df: pd.DataFrame) -> None:
    """特征表的最小契约。缺一项就抛，不猜默认值。"""
    if df is None or len(df) == 0:
        raise DataContractError("特征表为空")
    missing = [c for c in (CELL_COL, CYCLE_COL, SOH_COL) if c not in df.columns]
    if missing:
        raise DataContractError(f"特征表缺列：{missing}；至少需要 {CELL_COL}/{CYCLE_COL}/{SOH_COL}")
    feats = feature_columns(df)
    if not feats:
        raise DataContractError(f"特征表没有任何 {FEATURE_PREFIX}<i> 列")
    if df[CELL_COL].isna().any():
        raise DataContractError(f"{CELL_COL} 列有空值：电池 ID 缺失会让划分静默漏掉数据")
    if df[SOH_COL].isna().any():
        raise DataContractError(f"{SOH_COL} 列有空值：标签缺失会让 loss 变 nan")


def load_feature_table(path: str | Path) -> pd.DataFrame:
    """读特征表。支持 ``.csv`` / ``.parquet`` / ``.json``。

    返回的 DataFrame 一定通过 :func:`validate_feature_table`。
    """
    path = Path(path)
    if not path.exists():
        raise DataContractError(f"特征表不存在：{path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".json":
        df = pd.read_json(path)
    else:
        raise DataContractError(f"不支持的特征表格式 {suffix}（支持 .csv/.parquet/.json）")
    validate_feature_table(df)
    return df


@dataclass(frozen=True)
class FeatureTable:
    """特征表的只读视图：暴露维度与电池清单，不暴露可变状态。"""

    frame: pd.DataFrame
    feats: tuple[str, ...]

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "FeatureTable":
        validate_feature_table(df)
        return cls(frame=df.reset_index(drop=True), feats=tuple(feature_columns(df)))

    @property
    def input_dim(self) -> int:
        return len(self.feats)

    @property
    def cells(self) -> list[str]:
        return sorted(self.frame[CELL_COL].unique().tolist())

    def subset(self, cell_ids) -> pd.DataFrame:
        wanted = set(cell_ids)
        out = self.frame[self.frame[CELL_COL].isin(wanted)]
        return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# ΔQ(V) 特征
# ---------------------------------------------------------------------------


def delta_q_curve(
    voltage: np.ndarray,
    capacity: np.ndarray,
    v_grid: np.ndarray,
) -> np.ndarray:
    """把一次放电的 (V, Q) 插值到固定电压网格上，得到 Q(V)。

    Severson et al. 2019 的特征构造方式：放电容量对电压作图，在统一网格上
    重采样，使得不同电池、不同循环长度可以逐点相减。

    ``voltage`` 必须单调（升或降都可以，内部会排序）。``v_grid`` 必须落在
    ``voltage`` 的值域内，否则线性外延会产生没有物理意义的值——这里直接抛错，
    不外延。
    """
    voltage = np.asarray(voltage, dtype=float)
    capacity = np.asarray(capacity, dtype=float)
    v_grid = np.asarray(v_grid, dtype=float)
    if voltage.shape != capacity.shape:
        raise DataContractError(f"voltage 与 capacity 长度不一致：{voltage.shape} vs {capacity.shape}")
    if voltage.size < 2:
        raise DataContractError("至少需要 2 个采样点才能插值")
    lo, hi = float(voltage.min()), float(voltage.max())
    if v_grid.min() < lo - 1e-9 or v_grid.max() > hi + 1e-9:
        raise DataContractError(
            f"电压网格 [{v_grid.min():.4f}, {v_grid.max():.4f}] 超出数据值域 "
            f"[{lo:.4f}, {hi:.4f}]；拒绝外延，请改网格或改区间"
        )
    order = np.argsort(voltage)
    return np.interp(v_grid, voltage[order], capacity[order])


def delta_q_features(
    cycles: list[tuple[np.ndarray, np.ndarray]],
    v_grid: np.ndarray,
    reference_index: int = 0,
) -> np.ndarray:
    """ΔQ(V) = 当前循环 Q(V) − 参考循环 Q(V)，逐点相减。

    ``cycles`` 是同一颗电池按时间排序的 (voltage, capacity) 序列，
    ``reference_index`` 指定用哪个循环作基准（0 = 最早那个）。

    返回 ``[len(cycles) - 1, len(v_grid)]``：参考循环自身不产生特征。
    第一维是循环号，第二维是电压网格——展平后就是 plan.md §4.1 的
    1000 维输入（如果 ``len(cycles) - 1 == 1`` 且 ``len(v_grid) == 1000``）。
    """
    if not cycles:
        raise DataContractError("cycles 为空")
    if not 0 <= reference_index < len(cycles):
        raise DataContractError(f"reference_index={reference_index} 越界（共 {len(cycles)} 个循环）")
    curves = [delta_q_curve(v, c, v_grid) for v, c in cycles]
    ref = curves[reference_index]
    keep = [i for i in range(len(curves)) if i != reference_index]
    return np.stack([curves[i] - ref for i in keep], axis=0)


def make_v_grid(v_lo: float, v_hi: float, n_points: int) -> np.ndarray:
    """均匀电压网格。Severson 用 100–10% 区间，本项目默认照抄。"""
    if not v_lo < v_hi:
        raise DataContractError(f"电压区间不合法：[{v_lo}, {v_hi}]")
    if n_points < 2:
        raise DataContractError(f"网格点数至少 2，收到 {n_points}")
    return np.linspace(v_lo, v_hi, n_points)


# ---------------------------------------------------------------------------
# 本地标准化
# ---------------------------------------------------------------------------


class LocalStandardizer:
    """按**单个客户端的训练电池**拟合的标准化器。

    ``fitted_on`` 是不可变的 frozenset，``make_client_loaders`` 会拿它断言
    "val 电池没参与拟合"。这条断言是 plan.md §14 坑 1 的唯一执行者。
    """

    def __init__(self, eps: float = 1e-8) -> None:
        if eps <= 0:
            raise DataContractError(f"eps 必须为正，收到 {eps}")
        self.eps = eps
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._fitted_on: frozenset[str] = frozenset()

    @property
    def fitted(self) -> bool:
        return self._mean is not None

    @property
    def fitted_on(self) -> frozenset[str]:
        return self._fitted_on

    def fit(self, table: FeatureTable, train_cell_ids) -> "LocalStandardizer":
        cells = frozenset(train_cell_ids)
        if not cells:
            raise DataContractError("拒绝用空电池集合拟合标准化器：那等于全局标准化")
        unknown = cells - set(table.cells)
        if unknown:
            raise DataContractError(f"要拟合的电池不在特征表里：{sorted(unknown)[:5]}")
        sub = table.subset(cells)
        feats = sub.loc[:, list(table.feats)].to_numpy(dtype=float)
        self._mean = feats.mean(axis=0)
        std = feats.std(axis=0)
        # 常数列（如全零的无效特征）不能把除零放大成 inf
        self._std = np.where(std < self.eps, 1.0, std)
        self._fitted_on = cells
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise DataContractError("标准化器尚未拟合")
        x = np.asarray(x, dtype=float)
        if x.shape[-1] != self._mean.shape[0]:
            raise DataContractError(
                f"特征维度 {x.shape[-1]} 与拟合时的 {self._mean.shape[0]} 不一致"
            )
        return (x - self._mean) / self._std

    def check_no_leak(self, forbidden_cell_ids, label: str = "评估集") -> None:
        """断言这些电池没有参与拟合。这是坑 1 的执行点。"""
        overlap = frozenset(forbidden_cell_ids) & self._fitted_on
        if overlap:
            raise DataContractError(
                f"{label}有 {len(overlap)} 颗电池参与了标准化拟合："
                f"{sorted(overlap)[:5]}。这是 plan.md §14 坑 1 禁止的全局标准化"
            )


class FixedScaler:
    """与数据无关的固定缩放。零泄露，推荐用于全局测试集。"""

    def __init__(self, scale: np.ndarray, shift: np.ndarray | None = None) -> None:
        self.scale = np.asarray(scale, dtype=float)
        self.shift = np.zeros_like(self.scale) if shift is None else np.asarray(shift, dtype=float)
        if np.any(self.scale == 0):
            raise DataContractError("FixedScaler 的 scale 含 0，会把特征除成 inf")

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[-1] != self.scale.shape[0]:
            raise DataContractError(f"特征维度 {x.shape[-1]} 与 scaler 的 {self.scale.shape[0]} 不一致")
        return (x - self.shift) / self.scale


# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------


class CellDataset(Dataset):
    """按电池 ID 过滤的 (features, soh) 数据集。

    只认 ``cell_id`` 集合，不做任何时间切分——时间切分的规则是
    plan.md §6.2 规则 1（以电池为最小原子，客户端内再按时间切 train/val），
    属于调用方的策略，不是加载器该决定的事。
    """

    def __init__(self, table: FeatureTable, cell_ids, scaler=None) -> None:
        wanted = list(dict.fromkeys(cell_ids))
        unknown = set(wanted) - set(table.cells)
        if unknown:
            raise DataContractError(f"电池不在特征表里：{sorted(unknown)[:5]}")
        sub = table.subset(wanted)
        self.cell_ids = wanted
        self.cycles = sub[CYCLE_COL].to_numpy()
        self.y = torch.tensor(sub[SOH_COL].to_numpy(dtype=float), dtype=torch.float32)
        x = sub.loc[:, list(table.feats)].to_numpy(dtype=float)
        if scaler is not None:
            x = scaler.transform(x)
        self.x = torch.tensor(x, dtype=torch.float32)
        self.input_dim = x.shape[1]

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


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
) -> ClientData:
    """构造一个客户端的 DataLoader 对（plan.md §3.2 的模块契约入口）。

    标准化器只拟合 ``train_cell_ids``，随后断言 ``val_cell_ids`` 没参与。
    ``shuffle`` 用派生自 ``seed`` 的 generator，不用全局随机——否则
    plan.md §2.3 的"固定 seed"是一条空话。
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

    train_ds = CellDataset(table, train_cells, scaler)
    val_ds = CellDataset(table, val_cells, scaler) if val_cells else None

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


def make_global_test_loader(
    table: FeatureTable,
    test_cell_ids,
    *,
    strategy: str = "test",
    fixed_scale: np.ndarray | None = None,
    batch_size: int = 64,
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

    ds = CellDataset(table, cells, scaler)
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
