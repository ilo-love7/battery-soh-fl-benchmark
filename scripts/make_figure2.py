#!/usr/bin/env python3
"""图 2：标签偏斜 → 精度（plan.md §13.3 改写后的 M3 DOD ③）。

**横轴是标签偏斜，不是 k。** 原 M3 DOD 写的是「α 四档」，但 α 控制的是数量
偏斜，而 M2 已证明 H2 的效应来自标签偏斜。所以这张图要回答的问题是：

    偏斜多大才开始掉点？掉多少？

横轴取「客户端身份解释的标签方差占比」（`between_client_variance_fraction`，
定义见 `src/flsoh/heterogeneity.py`）。这是一个有物理含义的量——它度量
「如果每个客户端只按自己的数据拟合，能解释全局标签方差的多少」——而不是
一个实现参数。

**必须同时标出 N 与 k，因为这两个量随偏斜一起变。** sohrank 的 k 从 10 降到 1
时，N 从 12 涨到 115、偏斜从 39.2% 涨到 70.4%。曲线因此有两个可能的解释：
「偏斜导致掉点」和「客户端越多掉点越多」。要分离它们，每一档都需要一个
**同 k 的 IID 对照**——同 k 保证 N 与聚合结构相同，只改「哪些电池分给谁」。
所以这张图画两条线：sohrank（实心）与同 k 的 IID（空心），两点之间的竖直
距离才是偏斜的净效应。

画法上的两个决定：

1. **不画趋势线/拟合线。** 只有 4–5 个点，拟合出来的斜率只是把噪声讲成
   规律。点与点之间画细虚线连接，只表示「同一策略下相邻档位」，不表示
   任何插值。
2. **误差棒是跨 seed 的标准差，不用 ±1σ 带。** n=8 时带的下界会被 sd 压穿
   （M2 的图 1 上见过这个假象），而误差棒至少是诚实的区间表示。

用法::

    python scripts/make_figure2.py                 # 写到 docs/figures/
    python scripts/make_figure2.py --out /tmp/x.png
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def load_runs(results_dir: Path) -> list[dict]:
    runs = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "final_mae" not in data:
            continue
        runs.append(data)
    return runs


def _key(run: dict) -> tuple:
    """把 run 归到 (strategy, scheme, k, local_epochs)。

    四个字段都必须进 key：

    - `strategy`：**FedProx 的 run 不能混进 FedAvg 的线**。dirichlet 划分
      上跑过 fedprox（两个 μ × 8 seeds），少了这一维它们会被并进
      「FedAvg / dirichlet」——图上不画这条线，但汇总表会把 n=8 报成
      n=24，而读者会以为那是 FedAvg 的结果。
    - `scheme`：sohrank / dirichlet / iid 三种划分要分开。**dirichlet 不能
      归到 iid**——它的偏斜（12.2%）与 IID（17.3%）不同，混在一起两条线
      都会错。
    - `k`：分片粗细，决定 N 与偏斜。
    - `local_epochs`：同一份划分跑过 le=3 和 le=10 两档（等训练量对照），
      MAE 差一个量级，混在一起会把均值拖到两者都不代表的数上。
      这是 M2 已经踩过的坑（§10.2 第 1 条）。
    """
    src = str(run.get("partition_json") or "")
    name = Path(src).name
    k = None
    for token in name.replace(".json", "").split("-"):
        if token.startswith("k") and token[1:].isdigit():
            k = int(token[1:])
    if "sohrank" in name:
        scheme = "sohrank"
    elif "dirichlet" in name:
        scheme = "dirichlet"
    else:
        scheme = "iid"
    return str(run.get("strategy", "?")), scheme, k, run.get("local_epochs")


def aggregate(runs: list[dict]) -> dict[tuple, list[float]]:
    out: dict[tuple, list[float]] = {}
    for r in runs:
        out.setdefault(_key(r), []).append(float(r["final_mae"]))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="画图 2：标签偏斜 → 精度")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--hetero", default=str(ROOT / "results" / "heterogeneity.json"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures" / "fig2-skew-mae.png"))
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hetero = json.loads(Path(args.hetero).read_text(encoding="utf-8"))
    skew_by_k: dict[int, dict[str, float]] = {}
    n_by_k: dict[int, dict[str, int]] = {}
    for name, rep in (hetero.get("partitions") or {}).items():
        k = None
        for token in name.replace(".json", "").split("-"):
            if token.startswith("k") and token[1:].isdigit():
                k = int(token[1:])
        if k is None:
            continue
        if "sohrank" in name:
            scheme = "sohrank"
        elif "dirichlet" in name:
            scheme = "dirichlet"
        else:
            scheme = "iid"
        skew_by_k.setdefault(k, {})[scheme] = rep["between_client_variance_fraction"]
        n_by_k.setdefault(k, {})[scheme] = rep["n_clients"]

    runs = load_runs(Path(args.results))
    groups = aggregate(runs)
    if not groups:
        print(f"FAIL {args.results} 里没有带 final_mae 的 run")
        return 1

    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    for scheme, colour, marker, label in (
        ("sohrank", "C3", "o", "sohrank shards (sorted by SOH)"),
        ("iid", "C0", "o", "IID, same k (control)"),
    ):
        pts = []
        for (strat, sch, k, le), maes in groups.items():
            if sch != scheme or k is None or len(maes) < 2:
                continue
            if strat != "fedavg":
                continue   # 这张图只画 FedAvg；FedProx 是另一条假设（H3）
            if le is not None and int(le) != 10:
                continue   # 只画默认口径；等训练量那档是另一条曲线
            skew = skew_by_k.get(k, {}).get(scheme)
            if skew is None:
                continue
            pts.append((
                skew * 100,
                statistics.mean(maes),
                statistics.stdev(maes),
                k,
                n_by_k.get(k, {}).get(scheme, 0),
                len(maes),
            ))
        if not pts:
            continue
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        es = [p[2] for p in pts]
        if scheme == "sohrank":
            ax.errorbar(xs, ys, yerr=es, color=colour, marker=marker, ms=7,
                        lw=1.4, capsize=3, zorder=3, label=label)
        else:
            ax.errorbar(xs, ys, yerr=es, color=colour, marker=marker, ms=7,
                        lw=1.4, capsize=3, mfc="white", zorder=3, label=label)
        # 相邻档位用细虚线连接，只表示"同一策略下相邻"，不表示插值
        ax.plot(xs, ys, color=colour, lw=0.7, ls=":", alpha=0.6, zorder=2)
        for x, y, _, k, n, cnt in pts:
            ax.annotate(f"k={k}\nN={n}", (x, y), textcoords="offset points",
                        xytext=(6, 6), fontsize=7.5, color=colour)

    # C0 参考线。不用 `axhline(..., label=...)`：本图 y 轴是线性的所以现在
    # 看不出来，但一旦改成对数轴，matplotlib 3.11.2 会把图例排乱（图 1 已经
    # 踩过，见 tests/test_figure_style.py）。用 get_xaxis_transform() 的
    # 普通 Line2D 是两种轴上都安全的写法。
    c0 = [r["final_mae"] for r in runs if r.get("strategy") == "centralized"]
    if c0:
        ax.plot([0, 1], [statistics.mean(c0)] * 2, color="k", ls="--", lw=1.1,
                zorder=1, transform=ax.get_xaxis_transform(),
                label=f"C0 centralized, 300 epochs (n={len(c0)})")

    ceiling = (hetero.get("ceiling") or {}).get("between_client_variance_fraction")
    if ceiling:
        ax.axvline(ceiling * 100, color="gray", ls="-.", lw=1.0, alpha=0.7, zorder=1)
        ax.annotate(f"skew ceiling {ceiling * 100:.1f}%\n(one cell per client)",
                    (ceiling * 100, ax.get_ylim()[1]),
                    textcoords="offset points", xytext=(-6, -14),
                    fontsize=7.5, color="gray", ha="right")

    ax.set_xlabel("label skew — fraction of label variance explained by client identity (%)")
    ax.set_ylabel("test MAE")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.set_title("M3: accuracy cost of label skew (same total data, same budget;\n"
                 "vertical gap between the two lines at a given k isolates the skew)",
                 fontsize=9.5)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"已写入 {out} 与 {out.with_suffix('.pdf')}")
    for (strat, sch, k, le), maes in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if k is None or len(maes) < 2:
            continue
        skew = skew_by_k.get(k, {}).get(sch)
        if skew is None:
            continue
        print(f"  k={k:<3} {sch:<8} N={n_by_k.get(k, {}).get(sch, 0):<4} "
              f"{strat:<8} le={le}  skew={skew * 100:5.1f}%  "
              f"MAE={statistics.mean(maes):.5f} "
              f"± {statistics.stdev(maes):.5f}  (n={len(maes)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
