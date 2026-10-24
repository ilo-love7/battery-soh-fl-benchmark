#!/usr/bin/env python3
"""把 results/ 里的 run JSON 汇总成 plan.md §10.2 的表 1。

为什么要脚本而不是手抄：表 1 的每个数都要能追溯到具体文件，而且换 seed 数、
换 α 之后要重出。手抄一次之后，第二次就会开始"差不多就行"，那是造假的开始。

相对劣化的分母是 **C0 中心化**（plan.md §11.3）。C0 缺失时直接报错，
不拿 C1 顶替——分母换了而没记录，比没有分母更糟。

用法::

    python scripts/summarize_results.py --results results/
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _load(results_dir: Path) -> list[dict]:
    runs = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # 划分产物也在同一目录，它们没有 final_mae 字段
        if "final_mae" not in data:
            continue
        data["_file"] = path.name
        runs.append(data)
    return runs


def _fmt(vals: list[float]) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.5f}"
    mean = statistics.mean(vals)
    std = statistics.stdev(vals)
    return f"{mean:.5f} ± {std:.5f}"


def _paired(runs: list[dict], a_prefix: str, b_prefix: str) -> None:
    """按 seed 配对比两个配置。

    为什么需要配对：跨 seed 的绝对波动（本仓库实测 13%–18%）远大于要检测的
    效应（5%），所以单看两组的 mean±std 判不出差异。配对差把「seed 带来的
    共同波动」消掉，只留配置差异——这是当前数据下唯一有分辨力的做法。
    """
    def by_seed(prefix: str) -> dict[int, float]:
        out = {}
        for r in runs:
            rid = str(r.get("run_id", ""))
            if rid.startswith(prefix):
                out[int(r.get("seed", -1))] = r["final_mae"]
        return out

    a, b = by_seed(a_prefix), by_seed(b_prefix)
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        print(f"  {a_prefix} vs {b_prefix}: 共同 seed 不足 2 个，跳过")
        return

    diffs = [a[s] - b[s] for s in common]
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    rel = abs(sd / mean) * 100 if mean else float("inf")
    neg = sum(1 for d in diffs if d < 0)
    verdict = "方向稳定" if neg in (0, len(diffs)) else "方向不一致"
    print(
        f"  {a_prefix} − {b_prefix}: 配对差均值 {mean:+.5f}  std {sd:.5f}  "
        f"相对波动 {rel:.1f}%  {neg}/{len(diffs)} 为负 → {verdict}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="汇总实验结果为表 1")
    ap.add_argument("--results", default="results")
    ap.add_argument("--baseline", default="c0__centralized",
                    help="相对劣化的分母，默认 C0 中心化")
    ap.add_argument("--paired", action="store_true",
                    help="额外输出按 seed 配对比较（噪声大时唯一有分辨力的做法）")
    args = ap.parse_args(argv)

    runs = _load(Path(args.results))
    if not runs:
        print(f"FAIL {args.results} 里没有带 final_mae 的 run")
        return 1

    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        key = (
            r.get("strategy", "?"),
            r.get("scheme") or "—",
            r.get("alpha"),
            r.get("local_epochs"),
            r.get("mu"),
        )
        groups.setdefault(key, []).append(r)

    base = [r for r in runs if str(r.get("run_id", "")).startswith(args.baseline)]
    if not base:
        print(
            f"FAIL 找不到基线 {args.baseline}*。相对劣化必须有分母——"
            "先跑 scripts/run_local_baseline.py"
        )
        return 2
    base_mae = statistics.mean(r["final_mae"] for r in base)
    print(f"基线 C0 中心化：MAE {base_mae:.5f}（{len(base)} 个 seed）\n")

    print("| 策略 | 划分 | α | local_epochs | μ | seeds | MAE (mean±std) | RMSE | 相对 C0 |")
    print("|---|---|---|---|---|---|---|---|---|")

    rows = []
    for key, items in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        strategy, scheme, alpha, le, mu = key
        maes = [r["final_mae"] for r in items]
        rmses = [r.get("final_rmse") for r in items if r.get("final_rmse") is not None]
        rel = (statistics.mean(maes) - base_mae) / base_mae * 100
        rows.append((strategy, scheme, alpha, le, mu, len(items), maes, rmses, rel))
        print(
            f"| {strategy} | {scheme} | {alpha if alpha is not None else '—'} "
            f"| {le if le is not None else '—'} | {mu if mu else '—'} "
            f"| {len(items)} | {_fmt(maes)} "
            f"| {_fmt([r for r in rmses])} | {rel:+.1f}% |"
        )

    # seed 波动：plan.md §11.3 要求相对标准差 <10%，超了要加 seed 数
    print("\nseed 波动检查（相对标准差，阈值 10%）：")
    bad = []
    for strategy, scheme, alpha, le, mu, n, maes, _, _ in rows:
        if n < 2:
            print(f"  {strategy}/{scheme}: 只有 {n} 个 seed，无法评估波动")
            continue
        mean = statistics.mean(maes)
        cv = statistics.stdev(maes) / mean * 100 if mean else float("inf")
        flag = "OK" if cv < 10 else "**超标**"
        if cv >= 10:
            bad.append((strategy, scheme, cv))
        print(f"  {strategy}/{scheme}: {cv:.1f}%  {flag}")

    if bad:
        print("\n超标项需要加 seed 数，不许挑最好的一次报（plan.md §11.3）。")

    if args.paired:
        print("\n配对比较（同 seed 相减，消掉 seed 共同波动）：")
        prefixes = sorted(
            {str(r.get("run_id", "")).rsplit("__seed", 1)[0] for r in runs}
        )
        for p in prefixes:
            if p == args.baseline:
                continue
            _paired(runs, p, args.baseline)
        # FedProx 的对照是 FedAvg，不是 C0
        for p in prefixes:
            if p.startswith("fedprox"):
                scheme = p.split("__", 1)[1] if "__" in p else ""
                _paired(runs, p, f"fedavg__{scheme}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
