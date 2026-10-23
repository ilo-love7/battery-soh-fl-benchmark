"""ΔQ(V) 特征与特征表契约（docs/plan.md §3.2 / §4.1）。

=========== ==========================================================
依赖         numpy + pandas。**不 import torch**
=========== ==========================================================

为什么单独一个模块
------------------
``build_features.py`` 要用这里的 ``delta_q_features`` / ``make_v_grid`` /
``validate_feature_table`` 生成特征表，而它跑在没有 torch 的环境里（CI 的
``matr`` job 刻意不装 torch，那是整个仓库最便宜的一道测试）。这些函数本身
一行 torch 代码都没有，没道理让它们被 torch 挡住。

所以拆成两层：本模块放**纯特征数学**，``flsoh.data`` 放 Dataset/DataLoader
并从这里 re-export——``from flsoh.data import CELL_COL`` 照旧能用。

规则写在仓库的第一条贡献约定里：**能被便宜测试覆盖的代码，不要被昂贵的依赖
挡住**。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


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
