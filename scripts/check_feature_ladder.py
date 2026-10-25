#!/usr/bin/env python3
"""在一份特征表上跑平凡基线阶梯，回答「信号在哪」。

    python3 scripts/check_feature_ladder.py --features data/mit-stanford/features.parquet

## 为什么需要这个脚本

``run_e2e.py`` 把阶梯接进了 e2e 的输出，但那只能看模拟数据。真实表上的
同一个问题（信号在幅度里还是在形状里）以前只能靠临时脚本回答，而临时脚本
的结论进不了文档——下次谁都得重做一遍。

实测（2026-10-25，117 颗 MATR 电池 × 20 循环，整颗留出 23 颗）：

    ====================== ========== ==========
    模型                    测试 MAE   相对常数
    ====================== ========== ==========
    常数基线                 0.00179    1.000
    max|ΔQ|（单标量）         0.00210    1.174
    median|ΔQ|              0.00204    1.140
    L2 范数                  0.00218    1.219
    全部 1000 维             0.00065    0.361
    ====================== ========== ==========

**每一个标量摘要都比常数基线差，而 1000 维全量好 2.8 倍。** 这条与模拟数据
上的结论**正好相反**（模拟数据：单标量 0.13 倍、1000 维 0.23 倍，形状是噪声）。
所以「模拟器缺形状机制」不是一句猜测：真实数据里 SOH 根本不在任何标量幅度
里，而在整条曲线的形状里；模拟器把它全放进了幅度。

## 读法

- ``只用幅度`` 明显不如 ``常数``：SOH 不在标量幅度里。
- ``全部特征`` 明显好于 ``只用幅度``：多维确实带增益。
- ``只用形状`` 明显不如 ``常数``：形状里也没有标签信息（幅度是唯一的载体）。

后两条同时成立是可能的，别把它当矛盾：``只用形状`` 把幅度除掉之后剩下的
东西可能与标签无关，而 ``全部特征`` 用的是幅度与形状的**组合**。

退出码：0 = 跑完；2 = 输入读不了。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from flsoh.baselines import (  # noqa: E402
    feature_ladder,
    ladder_verdict,
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

    print(f"\n[ladder] 结论：{ladder_verdict(L)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
