#!/usr/bin/env python3
"""逐 run 比较两个结果目录，回答「特征口径修复到底改了多少」。

    python3 scripts/compare_feature_versions.py \\
        --old results/ablation-old-features --new results/discharge-v2

## 为什么不能用 `--compare results/`

``results/`` 里那 172 个是**旧特征表 + 24 线程**跑出来的，新表是**新特征表 +
1 线程**。直接相减会把两件事混在一起：特征口径 + 线程数。线程数这一项单独就能
造成 |相对差| 中位 1.86%、最大 27.37%（``docs/plan.md`` §13.5.6）。所以比较的
对手必须是 ``results/ablation-old-features``——**旧特征表 + 1 线程**，那是专门
为隔离特征效应跑的 56 个 run。

脚本因此把「两边线程数必须相同」做成硬闸门，而不是提醒一句：口径不同的两个数
相减照样得到一个漂亮的比值，而那个比值没有含义。

## 它报什么

- 每个 run 的 旧/新 MAE 与比值
- 中位比值、四分位间距、范围、变好/变差个数
- **符号检验的精确二项 p 值**：比值落在 seed 噪声里时，只有方向是可数的
- 按配置家族分组的同一组数
- 每个家族在两个目录里的 **seed 变异系数**——判断「中位变化」是否超过噪声的
  唯一依据。中位比值 1.17 听起来像「劣化 17%」，但如果该家族的 seed CV 是
  60%，那这个 17% 什么也说明不了。

## 退出码

0 = 报告已给出（**不判对错**：这个脚本只描述差异，不判断修复是好是坏）；
2 = 输入不足或口径不一致（线程数不同、配对数太少、某个目录读不了）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from math import comb
from pathlib import Path

#: 少于这么多对就不给结论。3 对里 3 对同向的二项 p 是 0.25，说什么都不够。
MIN_PAIRS = 5


def load_runs(directory: Path) -> dict[str, dict]:
    """读出目录里所有实验产物（按 run_id）。坏 JSON 与无 ``final_mae`` 的跳过。"""
    runs: dict[str, dict] = {}
    if not directory.is_dir():
        raise FileNotFoundError(f"目录不存在：{directory}")
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or "final_mae" not in payload:
            continue
        runs[path.stem] = payload
    return runs


def family_of(run_id: str) -> str:
    """去掉 seed 后缀，剩下的当作「配置家族」。

    ``fedavg__iid__le3__nm0__seed3`` → ``fedavg__iid__le3__nm0``
    ``fedavg-le3-iid-seed3``         → ``fedavg-le3-iid``
    """
    for sep in ("__seed", "-seed"):
        head, found, _tail = run_id.rpartition(sep)
        if found and head:
            return head
    return run_id


def sign_test_p(better: int, worse: int) -> float:
    """符号检验的**精确**双侧 p 值（二项分布，p=0.5）。

    不用正态近似：n 常常只有 8–56，而近似在尾部偏差很大——恰好是判定要用到的
    那个区间。平局（比值恰为 1）不计入 n，与标准做法一致。
    """
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def seed_cv(runs: dict[str, dict], family: str) -> float | None:
    """同一家族内 MAE 的变异系数。少于 3 个 seed 不给数（2 个点的 std 没意义）。"""
    values = [r["final_mae"] for rid, r in runs.items()
              if family_of(rid) == family and isinstance(r.get("final_mae"), (int, float))]
    if len(values) < 3:
        return None
    mean = statistics.fmean(values)
    if mean == 0:
        return None
    return statistics.stdev(values) / mean


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="逐 run 比较两个结果目录（特征口径修复改了多少）"
    )
    ap.add_argument("--old", type=Path, required=True, help="旧口径目录（如 results/ablation-old-features）")
    ap.add_argument("--new", type=Path, required=True, help="新口径目录（如 results/discharge-v2）")
    ap.add_argument("--min-pairs", type=int, default=MIN_PAIRS,
                    help=f"少于这么多对就不给结论（默认 {MIN_PAIRS}）")
    ap.add_argument("--quiet", action="store_true", help="只打印汇总，不逐 run 列表")
    args = ap.parse_args(argv)

    try:
        old = load_runs(args.old)
        new = load_runs(args.new)
    except FileNotFoundError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 2

    print(f"旧口径 {args.old}：{len(old)} 个 run")
    print(f"新口径 {args.new}：{len(new)} 个 run")

    pairs: list[tuple[str, float, float]] = []
    skipped: list[str] = []
    for run_id in sorted(set(old) & set(new)):
        a, b = old[run_id], new[run_id]
        ta, tb = a.get("torch_num_threads"), b.get("torch_num_threads")
        if ta is not None and tb is not None and ta != tb:
            # 线程数是与 seed 同量级的独立随机源，混进来就没有「特征效应」可言
            skipped.append(f"{run_id}：线程数 {ta} vs {tb}")
            continue
        o, n = float(a["final_mae"]), float(b["final_mae"])
        if o <= 0 or n <= 0:
            skipped.append(f"{run_id}：MAE 非正（{o}, {n}）")
            continue
        pairs.append((run_id, o, n))

    if skipped:
        print(f"\n跳过 {len(skipped)} 对（口径不可比）：")
        for line in skipped[:10]:
            print(f"  {line}")

    if len(pairs) < args.min_pairs:
        print(
            f"\nFAIL 只有 {len(pairs)} 对可比 run，少于 {args.min_pairs} 对。"
            "少于这个数，方向（几对变好几对变差）本身就在噪声里，不给结论。",
            file=sys.stderr,
        )
        return 2

    ratios = [n / o for _, o, n in pairs]
    better = sum(1 for r in ratios if r < 1.0)
    worse = sum(1 for r in ratios if r > 1.0)
    tie = len(ratios) - better - worse

    if not args.quiet:
        print(f"\n{'run_id':44s} {'旧 MAE':>10s} {'新 MAE':>10s} {'新/旧':>7s}")
        for run_id, o, n in pairs:
            print(f"{run_id:44s} {o:10.5f} {n:10.5f} {n / o:7.2f}")

    q1, q3 = _quartiles(ratios)
    print(f"\n配对 {len(pairs)} 个 run（比值 = 新/旧）")
    print(f"  中位 {statistics.median(ratios):.3f}   四分位 [{q1:.3f}, {q3:.3f}]"
          f"   范围 [{min(ratios):.3f}, {max(ratios):.3f}]")
    print(f"  新表更好 {better} / 更差 {worse}" + (f" / 持平 {tie}" if tie else ""))
    print(f"  符号检验 p = {sign_test_p(better, worse):.4f}"
          f"（n={better + worse}，平局不计）")

    # 中位比值 vs seed 噪声：不并列给出这两个数，「劣化 17%」会被当成结论
    print(f"\n{'配置家族':32s} {'n':>3s} {'中位比值':>9s} {'旧 seed CV':>11s} {'新 seed CV':>11s}")
    families = sorted({family_of(rid) for rid, _, _ in pairs})
    for fam in families:
        sub = [(o, n) for rid, o, n in pairs if family_of(rid) == fam]
        med = statistics.median([n / o for o, n in sub])
        cv_old = seed_cv(old, fam)
        cv_new = seed_cv(new, fam)
        cv_o = f"{cv_old:.1%}" if cv_old is not None else "—"
        cv_n = f"{cv_new:.1%}" if cv_new is not None else "—"
        print(f"{fam:32s} {len(sub):3d} {med:9.3f} {cv_o:>11s} {cv_n:>11s}")

    print(
        "\n读法：**先看 seed CV 再看中位比值**。中位比值偏离 1 的幅度如果落在该家族"
        "的 seed CV 之内，那这个偏移就是噪声，不能读成「修复让结果变好/变差」。"
        "有分辨力的只有两件：符号检验的 p，以及中位比值与 seed CV 的量级差。"
    )
    return 0


def _quartiles(values: list[float]) -> tuple[float, float]:
    """线性插值分位数（与 numpy 默认一致，但这里不引入 numpy）。"""
    ordered = sorted(values)
    return _quantile(ordered, 0.25), _quantile(ordered, 0.75)


def _quantile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


if __name__ == "__main__":
    raise SystemExit(main())
