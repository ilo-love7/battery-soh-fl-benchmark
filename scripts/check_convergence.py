#!/usr/bin/env python3
"""判 H1–H4 之前先问一句：这条轨迹收敛了吗？

2026-10-26 查出：**判定不显著不全是「效应小」或「seed 少」，有一部分是
「训练还没停稳」**。156 个带曲线的联邦 run 里，**88 个在末 20% 窗口仍未收敛**；
末段四分位间距（IQR/中位）中位 0.6%–27%。对这些臂来说，`final_mae`、末段
均值、真值 是三个不同的数，谁进表取决于我们在第几轮停下——那不是一个关于
方法的结论。

最典型的三个（前两个直接决定 H1）：

| 家族 | 收敛 seed | 斜率/轮 | IQR | 后果 |
|---|---|---|---|---|
| `fedavg__iid`（H1 主臂 / H2 对照） | 2/8 | −0.29% | 12.7% | 与 C0 的差距随轮数继续变 |
| `fedavg-le3-iid`（H1 等预算臂） | 0/8 | −1.30% | 25.1% | CV 101%，这一格没有精度 |
| `fedavg__sohrank`（H2 处理臂） | **7/8** | +0.01% | 0.8% | 处理臂收敛、对照臂不收敛 |

最后一行是 H2 的方向性偏置：**处理臂收敛、对照臂不收敛**。未收敛的对照臂
MAE 偏高（它还在降），于是处理组显得更好，实测的「偏斜代价 +3.4%」是真实
代价的**下界**。换句话说，「H2 未判定」这个结论偏保守——数据没有支持
「没有效应」，只是在有偏于「看不见效应」的方向上没测出来。

判据（两条同时满足才算收敛）：

- **斜率**：末 ``--window`` 轮的最小二乘斜率，归一化到「每轮相对变化」，
  ``|slope| <= --slope-tol``（默认 0.2%/轮）
- **抖动**：末 ``--window`` 轮的 ``(max−min)/mean <= --spread-tol``（默认 10%）

只满足斜率不满足抖动，说明它在平台附近乱走（`final_mae` 是抽签）；
只满足抖动不满足斜率，说明它稳但在稳着下降（`final_mae` 偏低）。

    python scripts/check_convergence.py --results results
    python scripts/check_convergence.py --results results --fail-on-unconverged
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

WINDOW_FRAC = 0.2
MIN_WINDOW = 10
SLOPE_TOL = 0.002
SPREAD_TOL = 0.10


def family_of(run_id: str) -> str:
    """切掉 seed 后缀。``seed3`` 这种以 seed 开头的 id 不能被切空。"""
    for sep in ("__seed", "-seed"):
        i = run_id.find(sep)
        if i > 0:
            return run_id[:i]
    return run_id


def window_for(n: int, frac: float = WINDOW_FRAC, minimum: int = MIN_WINDOW) -> int:
    """窗口按曲线长度的固定比例取，不按固定点数。

    C0 每 5 epoch 记一个点（300 epoch → 60 点），联邦每轮一个点（100 轮 →
    100 点）。用「末 20 点」会让 C0 的窗口覆盖末 100 epoch（三分之一训练）
    而联邦只覆盖末 20 轮——同一把尺子量出两套含义，C0 会显得比实际更不收敛。
    """
    return max(minimum, int(round(frac * n)))


def _quantile(xs: list[float], q: float) -> float:
    """线性插值分位数（不引入 numpy，与 baselines._quantile 同口径）。"""
    s = sorted(xs)
    if not s:
        raise ValueError("空序列没有分位数")
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def robust_slope(tail: list[float]) -> float:
    """半窗中位差 / 半窗长度，归一化成「每轮相对变化」。

    为什么不用最小二乘：C0 的轨迹是**尖峰型**的（seed2 在 epoch 120 冲到
    0.0249，平时 0.0025）。一个 10 倍的离群点能把 LS 斜率拉反号，而它只
    说明「那一轮恰好踩到坏点」，不说明趋势。中位数对单个离群点免疫。
    """
    n = len(tail)
    if n < 4:
        return 0.0
    half = n // 2
    first = statistics.median(tail[:half])
    second = statistics.median(tail[half:])
    overall = statistics.median(tail)
    if not overall:
        return 0.0
    return (second - first) / half / overall


def diagnostics(maes: list[float], frac: float = WINDOW_FRAC) -> dict:
    """一条曲线的收敛诊断。返回的 ``converged`` 由两条判据共同决定。

    两条判据都用稳健统计量（中位 / IQR），理由同 :func:`robust_slope`：
    尖峰轨迹上 ``(max−min)/mean`` 会到 95%，而那个 95% 只反映「有一个坏
    点」，不反映「没收敛」。
    """
    window = window_for(len(maes), frac)
    if len(maes) < window:
        return {"n": len(maes), "window": window, "slope": 0.0, "spread": 0.0,
                "converged": False, "reason": "曲线不足一个窗口"}
    tail = maes[-window:]
    median = statistics.median(tail)
    spread = ((_quantile(tail, 0.75) - _quantile(tail, 0.25)) / median
              if median else 0.0)
    slope = robust_slope(tail)
    still_descending = slope < -SLOPE_TOL
    too_jittery = spread > SPREAD_TOL
    reasons = []
    if still_descending:
        reasons.append(f"仍在降 {slope:+.3%}/轮")
    if too_jittery:
        reasons.append(f"末段 IQR {spread:.1%}")
    return {
        "n": len(maes),
        "window": window,
        "slope": slope,
        "spread": spread,
        "converged": not (still_descending or too_jittery),
        "reason": "、".join(reasons),
    }


def load_runs(results: Path) -> list[dict]:
    runs = []
    for path in sorted(results.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "final_mae" not in data:
            continue
        maes = [float(p["mae"]) for p in (data.get("curve") or []) if "mae" in p]
        data["_maes"] = maes
        runs.append(data)
    return runs


def build_report(runs: list[dict], frac: float) -> list[dict]:
    by_fam: dict[str, list[dict]] = {}
    for run in runs:
        if not run["_maes"]:
            continue
        rid = run.get("run_id") or "?"
        by_fam.setdefault(family_of(rid), []).append(run)

    rows = []
    for fam in sorted(by_fam):
        items = by_fam[fam]
        diags = [diagnostics(r["_maes"], frac) for r in items]
        finals = [r["final_mae"] for r in items]
        tails = [statistics.mean(r["_maes"][-10:]) for r in items]
        rows.append({
            "family": fam,
            "n": len(items),
            "converged": sum(1 for d in diags if d["converged"]),
            "slope": statistics.median(d["slope"] for d in diags),
            "spread": statistics.median(d["spread"] for d in diags),
            "final": statistics.median(finals),
            "tail": statistics.median(tails),
            "metric_gap": (statistics.median(tails) - statistics.median(finals))
                          / statistics.median(finals) if statistics.median(finals) else 0.0,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="判定前的收敛闸门")
    ap.add_argument("--results", default="results")
    ap.add_argument("--window-frac", type=float, default=WINDOW_FRAC,
                    help="窗口占曲线长度的比例（默认 0.2，最少 10 个点）")
    ap.add_argument("--slope-tol", type=float, default=SLOPE_TOL,
                    help="每轮相对变化的容忍度（默认 0.2％）")
    ap.add_argument("--spread-tol", type=float, default=SPREAD_TOL,
                    help="末段四分位间距 / 中位的容忍度（默认 10％）")
    ap.add_argument("--fail-on-unconverged", action="store_true",
                    help="有未收敛的家族就退出 1（CI 用）")
    args = ap.parse_args(argv)

    runs = load_runs(Path(args.results))
    rows = build_report(runs, args.window_frac)
    if not rows:
        print(f"FAIL {args.results} 里没有带 curve 的 run")
        return 2

    print(f"判据：末 {args.window_frac:.0%} 的曲线（至少 10 点）"
          f"|斜率| ≤ {args.slope_tol:.2%}/轮 且抖动 ≤ {args.spread_tol:.0%}\n")
    print(f"{'家族':<34s} {'收敛':>5s} {'斜率/轮':>9s} {'IQR':>7s} "
          f"{'末轮':>9s} {'末10均':>9s} {'口径差':>8s}")
    for r in rows:
        flag = f"{r['converged']}/{r['n']}"
        print(f"{r['family']:<34s} {flag:>5s} {r['slope']:>+8.3%} {r["spread"]:>6.1%} "
              f"{r['final']:>9.5f} {r['tail']:>9.5f} {r['metric_gap']:>+7.1%}")

    bad = [r for r in rows if r["converged"] < r["n"]]
    print(f"\n{len(rows)} 个家族里 {len(bad)} 个有未收敛的 seed；"
          f"共 {sum(r['n'] - r['converged'] for r in bad)} 个 run 未收敛")
    if bad:
        print("未收敛的家族**不能**参与判定：`final_mae` 与末段均值的差就是"
              "「停在哪一轮」的代价，换个口径结论就换。")
    if args.fail_on_unconverged and bad:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
