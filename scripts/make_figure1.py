#!/usr/bin/env python3
"""图 1：轮数–精度曲线（plan.md §10.1 的图 1）。

三个 panel 分别对应 H1 / H2 / H3，一个假设一个图：

- **panel 1（H1）**：同一种划分、两种训练量。它回答「联邦差是不是训练量
  不够」，而不是「联邦差不差」。
- **panel 2（H2）**：三种 FedAvg、**同预算同总量，只改哪些电池分给谁**。
  这一格是本轮的核心：Dirichlet α=0.5 造出的标签偏斜（12.2%）比 IID
  （17.3%）还小，所以它测不到 H2；SOH 排序分片把偏斜拉到 47.1%，才是
  H2 的真正处理。
- **panel 3（H3）**：同划分下 FedAvg vs 两个 μ 的 FedProx。

画法上三个决定，都是为了不让图比数据更自信：

1. **每个 seed 画一条细线，均值画粗线；不画 ±1σ 带。** 第一版画的是
   ±1σ，结果 n=3 时某轮 sd 大于 mean，下界被 `max(1e-6, ·)` 夹成一条
   掉到 1e-6 的尖刺——图上看起来像"某轮 MAE 近乎为零"，其实只是
   `fill_between` 在 sd 跨过均值时的假象。seed 少的时候，把每个 seed
   都画出来比任何汇总量都诚实。
2. **C0 画成水平带，不画曲线。** C0 的 `curve` 字段是空的——它是一次性
   300 epoch 训练，没有"轮"这个概念。硬给它编一条 x 轴会让读者以为它和
   联邦是同一个时间尺度。
3. **y 轴取对数。** MAE 从 0.03 降到 0.001，线性轴上前 20 轮就贴底了，
   正好把最有信息量的收敛尾部压扁。

用法::

    python scripts/make_figure1.py                 # 写到 docs/figures/
    python scripts/make_figure1.py --out /tmp/x.png
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
        data["_file"] = path.name
        runs.append(data)
    return runs


def curve_band(items: list[dict]) -> dict[int, dict]:
    """把同配置的多个 seed 合成「每轮的均值」。

    只保留**所有 seed 都跑到**的轮次：某个 seed 少跑一轮时，那一轮的均值
    会建立在不同数量的样本上，画出来是个看不见的台阶。
    """
    per_round: dict[int, list[float]] = {}
    for r in items:
        for point in r.get("curve") or []:
            per_round.setdefault(int(point["round"]), []).append(float(point["mae"]))
    n_seeds = len(items)
    out = {}
    for rnd, vals in sorted(per_round.items()):
        if len(vals) < n_seeds:
            continue
        out[rnd] = {
            "mean": statistics.mean(vals),
            "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }
    return out


def seed_curves(items: list[dict]) -> dict[int, dict[int, float]]:
    """每个 seed 各自的 (轮 → MAE)。"""
    out = {}
    for r in items:
        seed = int(r.get("seed", -1))
        out[seed] = {
            int(p["round"]): float(p["mae"]) for p in (r.get("curve") or [])
        }
    return out


def _key(run: dict) -> tuple:
    return (
        str(run.get("strategy")),
        run.get("scheme") or "-",
        run.get("alpha"),
        float(run.get("mu") or 0.0),
        run.get("local_epochs") if run.get("local_epochs") is not None
        else run.get("epochs"),
    )


def group(runs: list[dict]) -> dict[tuple, list[dict]]:
    g: dict[tuple, list[dict]] = {}
    for r in runs:
        g.setdefault(_key(r), []).append(r)
    return g


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="画图 1：轮数–精度曲线")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures" / "fig1-rounds-mae.png"))
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_runs(Path(args.results))
    if not runs:
        print(f"FAIL {args.results} 里没有带 final_mae 的 run")
        return 1
    groups = group(runs)

    c0 = [k for k in groups if k[0] == "centralized"]
    if not c0:
        print("FAIL 找不到 C0 中心化基线，图 1 的参考线画不出来")
        return 2
    c0_maes = [r["final_mae"] for r in groups[c0[0]]]
    c0_mean = statistics.mean(c0_maes)
    c0_sd = statistics.stdev(c0_maes) if len(c0_maes) > 1 else 0.0

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17.0, 4.8), sharey=True)

    # ---- panel 1：H1，训练量口径 ----
    left = [
        (("fedavg", "iid", None, 0.0, 10), "FedAvg / IID, 10 local epochs", "C0"),
        (("fedavg", "iid", None, 0.0, 3), "FedAvg / IID, 3 local epochs (budget-matched)", "C2"),
    ]
    # ---- panel 2：H2，标签偏斜。三者都是 FedAvg、同预算，只改「哪些电池给谁」 ----
    middle = [
        (("fedavg", "iid", None, 0.0, 10), "FedAvg / IID (skew 17.3%)", "C0"),
        (("fedavg", "dirichlet", 0.5, 0.0, 10), "FedAvg / Dirichlet α=0.5 (skew 12.2%)", "C1"),
        (("fedavg", "sohrank", None, 0.0, 10), "FedAvg / SOH-ranked shards (skew 47.1%)", "C4"),
    ]
    # ---- panel 3：H3，FedProx ----
    right = [
        (("fedavg", "dirichlet", 0.5, 0.0, 10), "FedAvg / Dirichlet α=0.5", "C0"),
        (("fedprox", "dirichlet", 0.5, 0.01, 10), "FedProx μ=0.01 / Dirichlet α=0.5", "C4"),
        (("fedprox", "dirichlet", 0.5, 0.1, 10), "FedProx μ=0.1 / Dirichlet α=0.5", "C1"),
    ]

    def draw(ax, specs):
        for key, label, colour in specs:
            if key not in groups:
                continue
            items = groups[key]
            band = curve_band(items)
            if not band:
                continue
            xs = sorted(band)
            # 先画每个 seed 的细线：seed 少的时候这比任何汇总量都诚实
            for seed, pts in sorted(seed_curves(items).items()):
                sx = [x for x in xs if x in pts]
                if not sx:
                    continue
                ax.plot(sx, [pts[x] for x in sx], color=colour, lw=0.7,
                        alpha=0.32, zorder=1)
            ax.plot(xs, [band[x]["mean"] for x in xs], color=colour, lw=2.0,
                    zorder=3, label=f"{label} (n={band[xs[0]]['n']}, mean)")
        # C0 参考带：它没有「轮」，画成水平带。
        # **不要用 axhline(..., label=...)**：axhline 用的是混合变换
        # （x 轴分数、y 数据），而 y 轴是对数的。2026-10-25 实测
        # matplotlib 3.11.2 在这种情况下会把图例排乱——条目顺序错、
        # 句柄画成横贯整个坐标区的长线。改用 get_xaxis_transform() 的
        # 普通 Line2D：横向上还是满宽，但句柄在图例里是正常的一小段。
        ax.plot([0, 1], [c0_mean, c0_mean], color="k", ls="--", lw=1.2,
                zorder=2, transform=ax.get_xaxis_transform(),
                label=f"C0 centralized, 300 epochs (n={len(c0_maes)})")
        ax.fill_between(
            [1, 100], [c0_mean - c0_sd] * 2, [c0_mean + c0_sd] * 2,
            color="k", alpha=0.10, lw=0, zorder=0,
        )
        ax.set_xlabel("federated round")
        ax.set_yscale("log")
        ax.grid(alpha=0.25, which="both", lw=0.5)
        ax.legend(fontsize=8, frameon=False)

    draw(ax1, left)
    draw(ax2, middle)
    draw(ax3, right)
    ax1.set_ylabel("test MAE (log scale)")
    # 标题**只描述画了什么，不下判定**。上一版把结论写进标题
    # （"H1: training budget dominates the gap" / "H2: label skew, not quantity
    # skew, is what costs accuracy" / "H3: FedProx is worse at every μ tested"），
    # 重跑之后三条全部变成「未判定」，而图看起来完全正常——这类过期比数字过期
    # 更危险，因为图会被单独引用。判定写在 docs/report.md 与 plan §11.3。
    ax1.set_title("H1: FedAvg / IID vs C0 centralized", fontsize=10)
    ax2.set_title("H2: FedAvg under label skew (sohrank) / quantity skew (Dirichlet)",
                  fontsize=10)
    ax3.set_title("H3: FedProx at μ=0.01 / 0.1 vs FedAvg, same partition",
                  fontsize=10)
    fig.suptitle(
        "MIT-Stanford, 135 cells (115 train / 20 test), N=23 clients — "
        "thin lines are individual seeds, thick line is their mean",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"已写入 {out} 与 {out.with_suffix('.pdf')}")
    print(f"C0 参考：MAE {c0_mean:.5f} ± {c0_sd:.5f}（{len(c0_maes)} seeds）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
