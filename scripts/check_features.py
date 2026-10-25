#!/usr/bin/env python3
"""特征表健康检查：每行 ΔQ(V) 的总变差必须在物理量级内。

    python3 scripts/check_features.py data/mit-stanford/features.parquet

## 为什么需要这个脚本

ΔQ(V) 是两条光滑 Q(V) 曲线之差，所以它**本身必须是光滑的**：总变差
（相邻点绝对值之和）与曲线幅度同量级。实测正确的放电段 ΔQ：

    ==================== ======= ======
    指标                  数值     量级
    ==================== ======= ======
    曲线幅度（max−min）     0.031    Ah
    总变差 TV               0.21    Ah
    ==================== ======= ======

TV 只比幅度大一个量级不到，这就是「光滑」的可执行判据。

## 它拦的是什么

有一类错误**不抛异常、不改形状、不改长度**，只在数值里留下痕迹：
把不相干的采样点配对到一起。2026-10-25 抓到的那个就是这样——MATR 每个
循环的数组是「充电 + 放电」拼起来的，曾经把整条循环喂给 ``delta_q_curve``，
它的 ``argsort`` 把放电段排进充电段的电压区间，而放电段的 ``Qc`` 是常数。
结果：

    ==================== ======= ====== =========
    喂给 delta_q_curve     TV     std    与正确值相关
    ==================== ======= ====== =========
    整条循环（曾经的代码） 269.60  0.3918   −0.019
    仅放电段（正确）         0.21  0.0026    —
    ==================== ======= ====== =========

TV 差 1280 倍，相关系数 −0.02。**而那 2700 行特征照样训出了 MAE 0.0016
的模型**——精度曲线完全正常，只有 TV 这个量能指出数据是错的。

所以这个脚本不做「像不像 ΔQ」的主观判断，只算一个数，跟一个阈值比。

## 阈值怎么定

默认 ``--max-tv-ratio 20``：TV / 曲线幅度 ≤ 20。正确数据实测约 6.8
（0.21 / 0.031），留 3 倍余量。比值而不是绝对值，因为幅度随电池而变。

退出码：0 = 全部通过；1 = 有行超限（打印最差的几行）；2 = 输入读不了。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from flsoh.features import (  # noqa: E402
    CELL_COL,
    CYCLE_COL,
    SOH_COL,
    DataContractError,
    curve_total_variation,
)

#: TV / 幅度 的上限。实测正确数据约 6.8，留 3 倍余量。
DEFAULT_MAX_TV_RATIO = 20.0


def check_feature_table(
    path: str | Path,
    *,
    max_tv_ratio: float = DEFAULT_MAX_TV_RATIO,
    top: int = 10,
) -> dict:
    """对特征表文件逐行算 TV/幅度，返回统计与超限行。

    只依赖 parquet 与 numpy，不 import torch——CI 的 ``matr`` job 刻意不装
    torch，而这道检查必须在每次生成特征表后都能跑。
    """
    path = Path(path)
    if not path.is_file():
        raise DataContractError(f"特征表不存在：{path}")
    table = pq.read_table(path)
    names = set(table.schema.names)
    for required in (CELL_COL, CYCLE_COL):
        if required not in names:
            raise DataContractError(f"特征表缺列 {required}（有 {sorted(names)[:6]}…）")
    feats = [c for c in table.schema.names
             if c not in (CELL_COL, CYCLE_COL, SOH_COL)]
    if not feats:
        raise DataContractError("特征表里一列特征都没有")

    # 一次性物化特征矩阵：1000 列 × 2700 行 × 8 B ≈ 21 MB，物化得起来。
    # 逐行去查 1000 个 column 反而慢一个量级。
    matrix = np.vstack([np.asarray(table.column(c).to_numpy(zero_copy_only=False),
                                   dtype=float) for c in feats]).T
    report = check_matrix(
        matrix,
        np.asarray(table.column(CELL_COL).to_pylist()),
        np.asarray(table.column(CYCLE_COL).to_pylist()),
        n_features=len(feats),
        max_tv_ratio=max_tv_ratio,
        top=top,
    )
    report["path"] = str(path)
    return report


def check_feature_frame(
    frame,
    *,
    max_tv_ratio: float = DEFAULT_MAX_TV_RATIO,
    top: int = 10,
) -> dict:
    """同一套判据，输入换成 DataFrame。

    端到端联调（``run_e2e.py``）里的表是内存里造出来的，不该为了跑一道检查
    先落一个 parquet 到磁盘——落盘再读回来，检查的就变成「写出去的那份」，
    而不是「真正要拿去训练的那份」。
    """
    names = list(frame.columns)
    for required in (CELL_COL, CYCLE_COL):
        if required not in names:
            raise DataContractError(f"特征表缺列 {required}（有 {names[:6]}…）")
    feats = [c for c in names if c not in (CELL_COL, CYCLE_COL, SOH_COL)]
    if not feats:
        raise DataContractError("特征表里一列特征都没有")
    matrix = np.asarray(frame[feats].to_numpy(dtype=float))
    report = check_matrix(
        matrix,
        frame[CELL_COL].to_numpy(),
        frame[CYCLE_COL].to_numpy(),
        n_features=len(feats),
        max_tv_ratio=max_tv_ratio,
        top=top,
    )
    report["path"] = "<memory>"
    return report


def check_matrix(
    matrix: np.ndarray,
    cells: np.ndarray,
    cycles: np.ndarray,
    *,
    n_features: int,
    max_tv_ratio: float = DEFAULT_MAX_TV_RATIO,
    top: int = 10,
) -> dict:
    """判据本体：逐行 TV/幅度。文件与 DataFrame 两条入口都走这里。"""
    n_rows = matrix.shape[0]
    if n_features < 2:
        raise DataContractError(f"只有 {n_features} 列特征，算不出总变差")

    ratios: list[tuple[float, str, int]] = []
    n_reference_rows = 0
    for i in range(n_rows):
        row = matrix[i]
        if not np.isfinite(row).all():
            # 含 NaN/inf：不是光滑，是坏数据。按超限处理。
            ratios.append((float("inf"), str(cells[i]), int(cycles[i])))
            continue
        if float(np.abs(row).max()) == 0.0:
            # 参考循环自身：ΔQ ≡ 0 是**构造出来的**，不是缺陷。
            # 单独计数，不参与 TV/幅度（幅度为 0，比值没有定义）。
            n_reference_rows += 1
            continue
        span = float(row.max() - row.min())
        if span <= 0:
            # 非零但整行常数：这才是「插值链把点配平了」的样子。
            ratios.append((float("inf"), str(cells[i]), int(cycles[i])))
            continue
        ratios.append((curve_total_variation(row) / span,
                       str(cells[i]), int(cycles[i])))

    bad = sorted([r for r in ratios if r[0] > max_tv_ratio], reverse=True)
    finite = [r[0] for r in ratios if np.isfinite(r[0])]
    n_cells = len(set(cells.tolist()))
    # 每颗电池最多一个参考循环。零行比电池还多，说明整张表退化成全零，
    # 而「全零」在 TV/幅度 上恰好是算不出比值的——不设这道闸门就会静默通过。
    too_many_zero_rows = n_reference_rows > n_cells

    # 诊断量（**不参与判定**）：相邻网格点的最大跳变 / 曲线幅度。
    #
    # 为什么需要它：TV/幅度 的噪声本底随「采样密度 × 电压噪声」线性增长——
    # 实验室路径（约 0.2 mV、5 s 采样）实测 5.9，而 App 路径（2 mV、10 s）
    # 实测中位 12.8、尾部 45。同一张表上错配数据的 TV/幅度 是 122，所以
    # 在 App 路径上 TV 的判别余量只有 2.7 倍，而不是实验室路径的 20 倍。
    #
    # 最大跳变对噪声不敏感（噪声是逐点小抖动，错配是**一处大跳变**）：
    # 错配 0.78、实验室正确 0.11、App 路径 0.19（尾部 0.55）。它作为
    # 第二判据更稳，但 0.55 与 0.78 的间隔不够宽，所以这里只报数不设闸门。
    steps: list[float] = []
    for i in range(n_rows):
        row = matrix[i]
        if not np.isfinite(row).all():
            continue
        span = float(row.max() - row.min())
        if span <= 0:
            continue
        steps.append(float(np.abs(np.diff(row)).max()) / span)

    return {
        "path": None,
        "n_rows": n_rows,
        "n_features": n_features,
        "max_tv_ratio": max_tv_ratio,
        "n_reference_rows": n_reference_rows,
        "n_cells": n_cells,
        "tv_ratio_median": float(np.median(finite)) if finite else float("nan"),
        "tv_ratio_max": float(np.max(finite)) if finite else float("inf"),
        "n_over_limit": len(bad),
        "too_many_zero_rows": too_many_zero_rows,
        "max_step_ratio_median": float(np.median(steps)) if steps else float("nan"),
        "max_step_ratio_max": float(np.max(steps)) if steps else float("nan"),
        "worst": [{"cell_id": c, "cycle": k, "tv_ratio": v} for v, c, k in bad[:top]],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="特征表健康检查：逐行核对 ΔQ(V) 的总变差",
    )
    parser.add_argument("table", help="特征表 parquet")
    parser.add_argument("--max-tv-ratio", type=float, default=DEFAULT_MAX_TV_RATIO,
                        help=f"TV/幅度 上限（默认 {DEFAULT_MAX_TV_RATIO}）")
    parser.add_argument("--top", type=int, default=10, help="最多打印几行超限样本")
    args = parser.parse_args(argv)

    try:
        report = check_feature_table(args.table, max_tv_ratio=args.max_tv_ratio,
                                     top=args.top)
    except DataContractError as exc:
        print(f"[check_features] 输入有问题：{exc}", file=sys.stderr)
        return 2

    print(f"[check_features] {report['n_rows']} 行 × {report['n_features']} 特征"
          f"（其中 {report['n_reference_rows']} 行是参考循环自身，ΔQ ≡ 0）")
    print(f"[check_features] TV/幅度 中位数 {report['tv_ratio_median']:.2f}，"
          f"最大 {report['tv_ratio_max']:.2f}，阈值 {report['max_tv_ratio']:.1f}")
    if report["too_many_zero_rows"]:
        print(f"[check_features] FAIL：{report['n_reference_rows']} 行全零，"
              f"而只有 {report['n_cells']} 颗电池。每颗电池最多一个参考循环——"
              "整表退化成全零时 TV/幅度 算不出比值，不拦就会静默通过。",
              file=sys.stderr)
        return 1
    if report["n_over_limit"] == 0:
        print("[check_features] PASS：每一行的 ΔQ(V) 都够光滑")
        return 0

    print(f"[check_features] FAIL：{report['n_over_limit']} 行的 ΔQ(V) 不够光滑。"
          "这不是「精度差一点」，是插值链上把不相干的点配到了一起——"
          "先查分段（MATR 每个循环含充电+放电两段）再查电压网格。", file=sys.stderr)
    for row in report["worst"]:
        print(f"    {row['cell_id']} 循环 {row['cycle']}: TV/幅度 = {row['tv_ratio']:.1f}",
              file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
