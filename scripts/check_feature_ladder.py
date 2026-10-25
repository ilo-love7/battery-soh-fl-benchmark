#!/usr/bin/env python3
"""在一份特征表上跑平凡基线阶梯，回答「信号在哪」。

    python3 scripts/check_feature_ladder.py --features data/mit-stanford/features.parquet

## 为什么需要这个脚本

``run_e2e.py`` 把阶梯接进了 e2e 的输出，但那只能看模拟数据。真实表上的
同一个问题（信号在幅度里还是在形状里）以前只能靠临时脚本回答，而临时脚本
的结论进不了文档——下次谁都得重做一遍。

实测（2026-10-25，重建完成后的 135 颗 MATR 电池 × 20 循环，整颗留出 23 颗，
5 个划分种子取中位）：

    ====================== ========== ==============
    模型                    测试 MAE   相对常数
    ====================== ========== ==============
    常数基线                 0.00205    1.000
    max|ΔQ|（单标量）         0.00271    1.28（1.09–1.32）
    median|ΔQ|              0.00224    1.09–1.45
    L2 范数                  0.00233    1.14–1.48
    全部 1000 维             0.00064    0.31（0.30–0.42）
    ====================== ========== ==============

**每一个标量摘要都比常数基线差（5 种子 × 6 种 = 30 个组合无一例外），而
1000 维全量好 2.4–3.4 倍。** 这条与模拟数据上的结论**正好相反**（模拟数据：
单标量 0.13 倍、1000 维 0.23 倍，形状是噪声）。

机制不是「SOH 藏在形状里」这种笼统说法。真实数据上幅度的**跨电池离散度
33.8%**，而标签的跨电池 std 只有 0.5%——幅度主要编码「这是哪颗电池」，在整颗
留出的划分下毫无用处，所以它连水平线都不如。模拟数据上同一乘性干扰项也在
（`Q_ref` CV 7.4%），但标签离散度 6.5%，同量级，所以 1 维就够了。**要复现真实
数据的阶梯表，模拟器得同时对齐标签离散度并真的带一个形状机制**，只做前者会
变成「幅度和形状都不行」——反的方向换一个，结论依然相反。

## 读法

- ``只用幅度`` 明显不如 ``常数``：SOH 不在标量幅度里。
- ``全部特征`` 明显好于 ``只用幅度``：多维确实带增益。
- ``只用形状`` 明显不如 ``常数``：形状单独也不够（幅度是唯一的载体）。
- ``[label]`` 那段把误差拆成组间/组内：**组间 = 这颗电池衰减得快不快，组内 =
  它现在衰减到哪**。不拆开，「好 3 倍」会被读成「能跟踪老化」——真实表上组内
  RMSE 比标签的组内 std 还大，模型根本没跟住循环间的变化。

退出码：0 = 跑完；2 = 输入读不了。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from flsoh.baselines import (  # noqa: E402
    amplitude,
    error_decomposition,
    feature_ladder,
    label_spread,
    ladder_verdict,
    pc_ladder,
    pc_verdict,
    ridge_predict,
    scalar_summaries,
)
from flsoh.features import CELL_COL, CYCLE_COL, SOH_COL  # noqa: E402
from flsoh.partition import split_test_cells  # noqa: E402


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    names = table.column_names
    feats = [c for c in names if c.startswith("f")]
    if not feats:
        raise ValueError(f"{path} 里没有 f0… 形式的特征列")
    X = np.column_stack([np.asarray(table.column(c).to_pylist(), dtype=float)
                         for c in feats])
    y = np.asarray(table.column(SOH_COL).to_pylist(), dtype=float)
    cells = np.asarray(table.column(CELL_COL).to_pylist())
    # 循环号只用来排序展示，不参与计算
    _ = np.asarray(table.column(CYCLE_COL).to_pylist())
    return X, y, cells, feats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="特征表的平凡基线阶梯")
    ap.add_argument("--features", type=Path, required=True, help="特征 parquet")
    ap.add_argument("--test-cells", type=int, default=None,
                    help="整颗留出的电池数（默认约 20％）")
    ap.add_argument("--split-seed", type=int, default=0,
                    help="留出哪些电池（默认 0；换一个看结论稳不稳）")
    args = ap.parse_args(argv)

    try:
        X, y, cells, feats = _load(args.features)
    except (OSError, ValueError) as exc:
        print(f"[ladder] FAIL 读不了 {args.features}：{exc}", file=sys.stderr)
        return 2

    uniq = sorted(set(cells.tolist()))
    n_test = args.test_cells if args.test_cells is not None else max(1, len(uniq) // 5)
    if n_test >= len(uniq):
        print(f"[ladder] FAIL 留出 {n_test} 颗但只有 {len(uniq)} 颗", file=sys.stderr)
        return 2
    train_c, test_c = split_test_cells(uniq, n_test, args.split_seed)
    train_mask = np.isin(cells, train_c)
    test_mask = np.isin(cells, test_c)

    print(f"[ladder] {len(uniq)} 颗 / {len(y)} 行 / {len(feats)} 维；"
          f"SOH [{y.min():.4f}, {y.max():.4f}]")
    print(f"[ladder] 留出 {len(test_c)} 颗（split seed {args.split_seed}）："
          f"训练 {int(train_mask.sum())} 行 / 测试 {int(test_mask.sum())} 行")

    L = feature_ladder(X, y, train_mask, test_mask)
    print(f"\n{'模型':24s} {'测试 MAE':>10s} {'相对常数':>9s}")
    for key, label in (
        ("constant", "常数基线"),
        ("amplitude", "只用幅度（1 维岭回归）"),
        ("all_features", f"全部 {L['n_features']} 维"),
        ("shape_only", "只用形状（归一化后）"),
    ):
        print(f"{label:24s} {L[key]:10.5f} {L[key] / L['constant']:9.3f}")

    # 把「幅度没用」与「我选的幅度统计量不好」分开
    print(f"\n{'一维摘要':24s} {'测试 MAE':>10s} {'相对常数':>9s}")
    for name, col in scalar_summaries(X).items():
        pred = ridge_predict(col[train_mask], y[train_mask], col[test_mask])
        mae = float(np.abs(pred - y[test_mask]).mean())
        print(f"{name:24s} {mae:10.5f} {mae / L['constant']:9.3f}")

    # 「好 3 倍」有两种完全不同的含义：估出**这颗电池**的水平（组间），还是
    # 跟住**它随循环**的变化（组内）。不拆开就会把前者读成后者。
    spread = label_spread(y[test_mask], cells[test_mask])
    print(
        f"\n[label] 标签离散度：组间 std {spread['between_std']:.5f}"
        f"（占方差 {spread['var_fraction_between']:.0%}），"
        f"组内 std 池化 {spread['within_std_pooled']:.5f} / "
        f"逐颗中位 {spread['within_std_median']:.5f}"
        f"（{spread['n_groups']} 颗）"
    )
    # 分母只能是**池化**的组内 std：下面的组内 RMSE 也是池化的。
    # 拿它去比「逐颗中位」在真实 MATR 表上差 4.8 倍，会把跟得住的模型
    # 判成「没跟住」——2026-10-25 就这么错过一次（plan §13.5.4）。
    print(
        f"{'模型':24s} {'组间 RMSE':>10s} {'组内 RMSE':>10s} {'组内/基准':>10s}"
    )
    amp_col = amplitude(X)
    for tag, pred in (
        ("常数基线", np.full(int(test_mask.sum()), y[train_mask].mean())),
        (
            "只用幅度（1 维岭回归）",
            ridge_predict(amp_col[train_mask], y[train_mask], amp_col[test_mask]),
        ),
        ("全部特征", ridge_predict(X[train_mask], y[train_mask], X[test_mask])),
    ):
        dec = error_decomposition(pred, y[test_mask], cells[test_mask])
        print(
            f"{tag:24s} {dec['between_rmse']:10.5f} {dec['within_rmse']:10.5f}"
            f"{dec['within_ratio']:10.3f}"
        )
    print(
        "[label] 组间 = 「这颗电池衰减得快不快」；组内 = 「它现在衰减到哪」。"
        "「组内/基准」= 组内 RMSE ÷ 池化组内 std："
        "<1 比每颗各猜一个常数更会跟循环，≈1 完全没跟住，>1 在放大噪声"
    )

    # 「1000 维」是假象还是真的用了 1000 维？主成分阶梯答这个。
    P = pc_ladder(X, y, train_mask, test_mask)
    print(f"\n{'主成分':22s} {'占方差':>8s} {'测试 MAE':>10s} {'相对常数':>9s}")
    for row in P["rows"]:
        print(f"前 {row['k']:2d} 个{row['explained_variance']:>13.1%}"
              f"{row['mae']:>12.5f}{row['ratio']:>11.3f}")
    print(f"[pca] {pc_verdict(P)}")

    print(f"\n[ladder] 结论：{ladder_verdict(L)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
