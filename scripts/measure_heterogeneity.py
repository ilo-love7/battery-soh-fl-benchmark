#!/usr/bin/env python3
"""测量一份划分到底造出了多少标签异质性。

用法：

    python scripts/measure_heterogeneity.py \
        --features data/mit-stanford/features.parquet \
        --partition-json results/partition-mvp-k5-dirichlet0.5.json

为什么需要这一步
----------------
H2 说「Non-IID 下 FedAvg 会显著掉点」。判 H2 之前必须先回答一个更前置的问题：
**这份划分真的造出了标签偏斜吗？** 如果所有客户端看到的 SOH 分布几乎一样，
那「FedAvg 没掉点」是数据没有可偏斜的东西，不是算法对异质性免疫——两者
的结论完全相反，而 MAE 表上看不出来。

输出三个量（定义见 `src/flsoh/heterogeneity.py`）：

- `between_client_variance_fraction`：客户端身份解释的标签方差占比。
  这是 H2 的**检验力上界**——它有多低，H2 最多就只能测到多弱的效应。
- `size_gini` / `size_ratio`：数量偏斜。数量偏斜会污染结论：一个占 18%
  数据的客户端会把 FedAvg 拉向「部分中心化」，于是 Non-IID 反而更像中心化。
- `cell_mean_std` vs `cell_within_std_median`：电池间 vs 电池内的标签离散度。
  电池内远小于电池间，说明模型主要要学「这颗电池的水平」，而这正是
  划分能偏斜的部分。

需要 pandas（读 parquet），所以只在装了 pandas 的环境跑，不进纯标准库 job。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flsoh.features import CELL_COL, SOH_COL, load_feature_table  # noqa: E402
from flsoh.heterogeneity import heterogeneity_report  # noqa: E402


def load_partition(path: str | Path) -> dict[str, list[str]]:
    """读划分产物。兼容两种形态：顶层就是 {client: [cells]}，或包一层 partition 键。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    part = data.get("partition", data)
    if not isinstance(part, dict):
        raise SystemExit(f"::error::{path} 不是合法的划分产物")
    return {str(k): [str(c) for c in v] for k, v in part.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="测量划分的标签异质性")
    ap.add_argument("--features", required=True, help="特征表 parquet")
    ap.add_argument("--partition-json", required=True, action="append",
                    help="划分产物，可重复传多份做对比")
    ap.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    ap.add_argument("--out", default=None,
                    help="把结果写成 JSON（判定脚本读这个文件做 manipulation check）")
    args = ap.parse_args(argv)

    df = load_feature_table(args.features)
    labels_by_cell: dict[str, list[float]] = {}
    for cell, sub in df.groupby(CELL_COL, sort=True):
        labels_by_cell[str(cell)] = [float(v) for v in sub[SOH_COL]]

    reports = {}
    for path in args.partition_json:
        part = load_partition(path)
        missing = {c for cells in part.values() for c in cells} - set(labels_by_cell)
        if missing:
            print(f"::error::{path} 引用了 {len(missing)} 个不在特征表里的电池",
                  file=sys.stderr)
            return 2
        reports[Path(path).name] = heterogeneity_report(part, labels_by_cell)

    if args.json:
        print(json.dumps(reports, indent=2, ensure_ascii=False))

    if args.out:
        ceiling = heterogeneity_report(
            {c: [c] for c in labels_by_cell}, labels_by_cell
        )
        payload = {
            "source": str(args.features),
            "partitions": reports,
            "ceiling": ceiling,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        print(f"\n已写出 {out}")

    if args.json:
        return 0

    for name, r in reports.items():
        print(f"\n=== {name} ===")
        print(f"  客户端 {r['n_clients']} 个 / 电池 {r['n_cells']} 颗")
        print(f"  全局 SOH std              {r['global_std']:.5f}")
        print(f"  电池间均值 std            {r['cell_mean_std']:.5f}")
        print(f"  电池内 std（中位）        {r['cell_within_std_median']:.5f}")
        print(f"  **客户端身份解释的方差**  {r['between_client_variance_fraction'] * 100:.1f}%")
        print(f"  样本量基尼                {r['size_gini']:.3f}  "
              f"(最大 {r['size_max']} / 最小 {r['size_min']} 颗"
              + (f"，{r['size_ratio']:.0f}×" if r["size_ratio"] else "") + ")")

    # 上限：把每颗电池单独当一个客户端（能造出的最大标签偏斜）
    ceiling = heterogeneity_report(
        {c: [c] for c in labels_by_cell}, labels_by_cell
    )
    print("\n=== 上界：一颗电池一个客户端（划分能造出的最大标签偏斜） ===")
    print(f"  客户端身份解释的方差      "
          f"{ceiling['between_client_variance_fraction'] * 100:.1f}%")
    print("\n读法：这个百分比是 H2 的检验力上界。它越低，「FedAvg 没掉点」越可能")
    print("是数据没有可偏斜的东西，而不是算法对异质性免疫。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
