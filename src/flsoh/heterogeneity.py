"""标签异质性的可测量定义（纯标准库）。

为什么需要这个模块
------------------
H2 假设「Non-IID 划分下 FedAvg 会显著掉点」。但要检验它，前提是**划分真的
造出了标签偏斜**。本仓库的 Dirichlet 划分按*电池*分配，它直接造成的是
**数量偏斜**（有的客户端 21 颗、有的 1 颗），是否同时造成**标签偏斜**要看
电池之间的 SOH 是否真的差得开——这是数据属性，不是划分算法能决定的。

2026-10-24 实测（MIT-Stanford，135 颗 × 20 循环）：

    SOH 全局 std            0.0058
    电池间均值 std          0.0049   （含离群电池 b2c1，它比其余低 0.046）
    去掉 b2c1 后            0.0029
    电池内 std（中位）      0.0007

电池内 std 只有电池间 std 的 1/4，说明 SOH 在单颗电池上几乎是个常数——
模型主要要学的是「这颗电池的 SOH 水平是多少」。那么**客户端身份最多能解释
多少标签方差**就成了硬上限：

    (0.0029 / 0.0058)² ≈ 25%

无论怎么划分都超不过这个数。这不是「划分不好」，是这批数据本身没有更多
可偏斜的东西。所以 H2 的检验力是有上界的，判 H2 时必须把这条写出来，
否则「H2 被推翻」会被误读成「联邦对 Non-IID 免疫」。

指标定义
--------
`between_client_variance_fraction`：
    把每个客户端的样本合起来算均值 μ_c，按样本数加权算这些 μ_c 的方差，
    再除以全局方差。等于「客户端身份解释了标签方差的百分之几」。
    0 表示所有客户端标签分布一致；1 表示每个客户端内部是常数、彼此完全不同。

`size_gini`：
    客户端样本量的基尼系数。0 = 完全均等，→1 = 极度不均。
    数量偏斜本身不是 H2 要测的东西，但它会污染结论：一个占 18% 数据的
    客户端会把 FedAvg 拉向「部分中心化」。
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _variance(xs: Sequence[float]) -> float:
    """总体方差（除以 n）。这里要的是「这批样本的离散程度」，不是对总体的估计。"""
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def between_client_variance_fraction(
    partition: Mapping[str, Sequence[str]],
    labels_by_cell: Mapping[str, Sequence[float]],
) -> float:
    """客户端身份能解释的标签方差占比。

    参数
    ----
    partition
        `{client_id: [cell_id, ...]}`，与划分产物同构。
    labels_by_cell
        `{cell_id: [soh, ...]}`，每颗电池的标签序列。

    返回
    ----
    [0, 1] 之间的浮点数。全局方差为 0 时返回 0.0（无方差可解释）。
    """
    all_labels: list[float] = []
    client_means: list[float] = []
    client_sizes: list[int] = []

    for cid in sorted(partition):
        cells = partition[cid]
        pooled: list[float] = []
        for cell in cells:
            if cell not in labels_by_cell:
                raise KeyError(f"客户端 {cid} 引用了不在标签表里的电池 {cell!r}")
            pooled.extend(float(v) for v in labels_by_cell[cell])
        if not pooled:
            raise ValueError(f"客户端 {cid} 没有任何标签，无法计算异质性")
        client_means.append(_mean(pooled))
        client_sizes.append(len(pooled))
        all_labels.extend(pooled)

    total_var = _variance(all_labels)
    if total_var == 0.0:
        return 0.0

    n_total = len(all_labels)
    between = sum(
        n * (m - _mean(all_labels)) ** 2 for n, m in zip(client_sizes, client_means)
    ) / n_total
    return between / total_var


def size_gini(sizes: Iterable[int]) -> float:
    """客户端样本量的基尼系数。

    用标准公式 `G = ΣΣ|x_i − x_j| / (2 n² x̄)`。空列表或全零返回 0.0。
    """
    xs = sorted(float(s) for s in sizes)
    n = len(xs)
    if n == 0:
        return 0.0
    total = sum(xs)
    if total == 0.0:
        return 0.0
    diffs = sum(abs(a - b) for a in xs for b in xs)
    return diffs / (2.0 * n * n * (total / n))


def heterogeneity_report(
    partition: Mapping[str, Sequence[str]],
    labels_by_cell: Mapping[str, Sequence[float]],
) -> dict:
    """一次算齐 H2 需要的所有量，返回可写进 JSON 的字典。"""
    sizes = [len(partition[c]) for c in sorted(partition)]
    all_labels = [float(v) for cell in labels_by_cell for v in labels_by_cell[cell]]
    cell_means = [_mean(list(labels_by_cell[c])) for c in sorted(labels_by_cell)]
    within = [
        _variance([float(v) for v in labels_by_cell[c]]) for c in sorted(labels_by_cell)
    ]
    return {
        "n_clients": len(partition),
        "n_cells": len(labels_by_cell),
        "global_std": _variance(all_labels) ** 0.5,
        "cell_mean_std": _variance(cell_means) ** 0.5,
        "cell_within_std_median": sorted(within)[len(within) // 2] ** 0.5,
        "between_client_variance_fraction": between_client_variance_fraction(
            partition, labels_by_cell
        ),
        "size_gini": size_gini(sizes),
        "size_max": max(sizes) if sizes else 0,
        "size_min": min(sizes) if sizes else 0,
        "size_ratio": (max(sizes) / min(sizes)) if sizes and min(sizes) else None,
    }
