#!/usr/bin/env python3
"""图 3：隐私-效用曲线（plan.md §10.1 的 M4 DOD ②）。

要回答的问题只有一句：

    本地 DP-SGD 把噪声加到多大，精度掉多少？掉到不能接受之前，ε 是多少？

## 为什么画两个面板

- **左：MAE vs noise_multiplier。** nm 是实验里真正拧的旋钮，也是复现时
  要照抄的参数。每个点旁边标 ε——因为 nm 本身不说明任何隐私强度，同样
  nm=1.0 在 600 步和 60 步下是两个完全不同的预算。
- **右：MAE vs ε。** 这才是字面意义上的「隐私-效用曲线」：横轴是隐私
  强度，纵轴是效用。plan §11.3 的判据（ε ≤ 8 时劣化 ≤30%）在这张图上
  是一条竖线加一条横线围出来的格子。

只画左面板会让读者自己去做 nm→ε 的换算，而那个换算恰恰是最容易出错
的一步（q 与步数都由训练协议钉死，不是自由变量）。

## 三条参考线，一条都不能少

1. **nm=0 机制对照**（只裁剪 + Poisson、不加噪）：没有它，曲线左端到
   非 DP 基线之间的落差会被读成「小噪声很便宜」，而那其实是 Poisson
   抽样换批大小 + 梯度裁剪的代价。plan §7 明确警告过这件事。
2. **非 DP 同预算基线**：DP 一共付出多少（含机制开销）。
3. **C0 中心化上限**：联邦掉的那部分与 DP 掉的那部分要能分开看。

## 画法决定

- **y 轴取对数。** nm 从 0 到 8，MAE 预期跨一个量级以上；线性轴会把
  小噪声那一段压成一条平线，而那一带正是判据所在的位置。
- **每个 seed 画一个淡点，均值画实心点 + 标准差误差棒。** 与图 1/图 2
  同一套约定：seed 少的时候，把每个 seed 都画出来比任何汇总量都诚实。
- **nm=0 不画在横轴上。** 它的 ε 是 ∞，对数轴上没有位置；画成水平参考
  线，并在图例里写明「无隐私保证」——把它画成一个点会让人以为
  「ε=∞ 也是一个 DP 档位」。

用法::

    python scripts/make_figure3.py                 # 写到 docs/figures/
    python scripts/make_figure3.py --out /tmp/x.png
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


def _scheme_of(run: dict) -> str:
    src = Path(str(run.get("partition_json") or "")).name
    if "sohrank" in src:
        return "sohrank"
    if "dirichlet" in src:
        return "dirichlet"
    return str(run.get("scheme") or "iid")


def _key(run: dict) -> tuple:
    """把 run 归到 (strategy, scheme, local_epochs, n_clients, nm, C)。

    六个字段都必须进 key，理由与图 2 同构——少一维不会让图崩，会让
    某一档的均值描述一个没跑过的配置：

    - ``strategy``：DP 与服务端优化器正交，但两者都改过就必须分开。
      M3 ② 已经为「FedProx 混进 FedAvg 组」付过一次代价（n=8 报成 24）。
    - ``scheme`` / ``local_epochs`` / ``n_clients``：划分与训练量口径。
    - ``noise_multiplier``：**主超参**。五档只差一个 nm，而 nm=1 与 nm=8
      的效用预期差一个量级。它还是 None/absent 与 0.0 的区分依据——
      前者是「完全没开 DP」，后者是「开了机制但不加噪的对照组」，
      这两件事在图上必须是两条不同的参考线。
    - ``max_grad_norm``（C）：同 nm 不同裁剪范数是两个实验——敏感度
      上界不同，ε 也不同（噪声 Std = nm × C / 批大小）。
    """
    return (
        str(run.get("strategy", "?")),
        _scheme_of(run),
        run.get("local_epochs"),
        run.get("n_clients"),
        run.get("noise_multiplier"),
        run.get("max_grad_norm"),
    )


def aggregate(runs: list[dict]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for r in runs:
        out.setdefault(_key(r), []).append(r)
    return out


def epsilon_of(run: dict) -> float | None:
    """ε；非 DP 的 run 返回 None（不是 0——0 会被读成"极强保证"）。"""
    if run.get("noise_multiplier") is None:
        return None
    eps = run.get("dp_epsilon")
    return None if eps is None else float(eps)


def pick_references(sel: dict[tuple, list[dict]],
                    arms: list[tuple]) -> tuple[list, list, str | None]:
    """选出两条参考线对应的组：`nm=0` 机制对照、非 DP 同预算基线。

    **必须与 DP 臂同策略、同划分、同 N。** 「同 le」只对齐了训练量，
    策略或划分不同就不是同一个实验。2026-10-25 第一次出图时这条没写，
    抓到的「非 DP 同预算」是同 le 的 **FedAdam**（0.00162）而不是
    **FedAvg**（0.01165）——参考线画低了 7 倍，而图的其他部分完全正常。
    这正是图 2 已经付过一次学费的那类事故：分组错了，输出照样画得出来。

    返回 `(zero, plain, error)`；`error` 非 None 时前两个是空表，
    调用方应当**拒绝出图**而不是少画一条线。
    """
    if not arms:
        return [], [], "没有加噪的 DP 臂，无法确定参考线的策略/划分/N"
    ref = (arms[0][0], arms[0][1], arms[0][3])

    def _matching(nm_test):
        return [k for k in sel
                if (k[0], k[1], k[3]) == ref and nm_test(k[4])]

    zero = _matching(lambda nm: nm == 0.0)
    plain = _matching(lambda nm: nm is None)
    missing = [n for n, got in (("nm=0 机制对照", zero),
                                ("非 DP 同预算基线", plain)) if not got]
    if missing:
        return [], [], ("缺参考线：" + "、".join(missing)
                        + f"。DP 臂是 strategy={ref[0]} scheme={ref[1]} N={ref[2]}")
    return zero, plain, None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="画图 3：隐私-效用曲线")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures" / "fig3-privacy-utility.png"))
    ap.add_argument("--le", type=int, default=None,
                    help="只画这一档 local_epochs；默认取 DP run 里出现的那一档")
    ap.add_argument("--eps-max", type=float, default=None,
                    help="ε 上限（默认取 configs/dp.toml 的 acceptance.dp_epsilon_max）")
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_runs(Path(args.results))
    groups = aggregate(runs)
    if not groups:
        print(f"FAIL {args.results} 里没有带 final_mae 的 run")
        return 1

    dp_keys = [k for k in groups if k[4] is not None]
    if not dp_keys:
        print(f"FAIL {args.results} 里没有 DP run（没有 noise_multiplier 字段）")
        print("  先跑：python scripts/run_federated.py --config configs/dp.toml "
              "--noise-multiplier <nm> ...")
        return 1

    le = args.le
    if le is None:
        les = {k[2] for k in dp_keys}
        if len(les) > 1:
            print(f"FAIL DP run 里有多个 local_epochs：{sorted(les)}。"
                  "训练量口径必须先用 --le 选定")
            return 1
        le = les.pop()
    eps_max = args.eps_max
    if eps_max is None:
        import tomllib

        with (ROOT / "configs" / "dp.toml").open("rb") as fh:
            eps_max = float(tomllib.load(fh)["acceptance"]["dp_epsilon_max"])

    # 只保留选定 le 的组
    sel = {k: v for k, v in groups.items() if k[2] == le}
    arms = sorted(
        (k for k in sel if k[4] is not None and k[4] > 0),
        key=lambda k: k[4],
    )
    if not arms:
        print(f"FAIL le={le} 下没有加噪的 DP run")
        return 1

    # 一帧图只能讲一个实验。DP 臂之间若策略/划分/N 不同，它们就不该画在
    # 同一组坐标里——混在一张图上会让读者以为差异全来自 nm。
    arms_ref = sorted({(k[0], k[1], k[3]) for k in arms})
    if len(arms_ref) > 1:
        print(f"FAIL le={le} 的 DP 臂不是同一个实验：{arms_ref}。"
              "策略/划分/N 不同的 DP run 要分开画（或用 --le 先分流）")
        return 1

    zero, plain, err = pick_references(sel, arms)
    if err:
        print(f"FAIL {err}")
        for c in sorted({(k[0], k[1], k[2], k[3], k[4]) for k in sel}):
            print(f"    strategy={c[0]} scheme={c[1]} le={c[2]} N={c[3]} nm={c[4]}")
        print("  参考线不是可选装饰：缺了它，DP 的代价会被读成噪声的代价。")
        return 1

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 5.2))

    # ---------------- 左：MAE vs noise_multiplier ----------------
    for k in arms:
        maes = [float(r["final_mae"]) for r in sel[k]]
        eps = epsilon_of(sel[k][0])
        mean, sd = statistics.mean(maes), (
            statistics.stdev(maes) if len(maes) > 1 else 0.0
        )
        nm = k[4]
        good = eps is not None and eps <= eps_max
        colour = "C2" if good else "C0"
        ax_l.errorbar([nm], [mean], yerr=[sd], color=colour, marker="o", ms=8,
                      capsize=4, lw=1.4, zorder=4,
                      label=f"DP (n={len(maes)})" if k is arms[0] else None)
        for r in sel[k]:
            ax_l.plot([nm], [float(r["final_mae"])], marker="o", ms=3.2,
                      color=colour, alpha=0.35, zorder=3)
        eps_txt = "ε=∞" if eps == float("inf") else (
            f"ε={eps:.4g}" if eps is not None else "ε=?")
        # 最右边那点的标注放左侧：右边界是数据自己顶出来的，标注在右侧会
        # 被画布裁掉半个字（2026-10-25 实测 ε=7.836 被切）。
        last = k is arms[-1]
        ax_l.annotate(eps_txt, (nm, mean), textcoords="offset points",
                      xytext=(-10 if last else 8, 6), fontsize=8.5,
                      color=colour, ha="right" if last else "left",
                      fontweight="bold" if good else "normal")
    # 三条水平参考线。**一律不用 `axhline(..., label=...)`**：axhline 用
    # 混合变换（x 轴分数、y 数据），而两个面板的 y 轴都是对数的。
    # 2026-10-25 实测 matplotlib 3.11.2 在这种情况下会把图例排乱——
    # 条目顺序错、句柄画成横贯整个坐标区的长线（图 1 已经踩过一次）。
    # 改用 get_xaxis_transform() 的普通 Line2D：横向仍然满宽，但图例
    # 里的句柄是正常的一小段。
    # nm=0 机制对照：不画点，因为它的 ε=∞ 在对数横轴上没有位置。
    tr_l = ax_l.get_xaxis_transform()
    if zero:
        maes = [float(r["final_mae"]) for r in sel[zero[0]]]
        ax_l.plot([0, 1], [statistics.mean(maes)] * 2, color="C1", ls="--",
                  lw=1.2, zorder=2, transform=tr_l,
                  label=f"nm=0 control: clipping+Poisson only, no guarantee "
                        f"(n={len(maes)})")
    if plain:
        maes = [float(r["final_mae"]) for r in sel[plain[0]]]
        ax_l.plot([0, 1], [statistics.mean(maes)] * 2, color="gray", ls=":",
                  lw=1.2, zorder=2, transform=tr_l,
                  label=f"no DP, same budget (n={len(maes)})")
    c0 = [float(r["final_mae"]) for r in runs if r.get("strategy") == "centralized"]
    if c0:
        ax_l.plot([0, 1], [statistics.mean(c0)] * 2, color="k", ls="-.",
                  lw=1.0, alpha=0.7, zorder=1, transform=tr_l,
                  label=f"C0 centralized (n={len(c0)})")
    ax_l.set_xscale("log")
    ax_l.set_yscale("log")
    ax_l.set_xticks([k[4] for k in arms])
    ax_l.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_l.set_xlabel("noise_multiplier (client-side DP-SGD)")
    ax_l.set_ylabel("test MAE")
    ax_l.grid(alpha=0.25, lw=0.5, which="both")
    ax_l.legend(fontsize=8, frameon=False, loc="upper left")
    ax_l.set_title(f"M4: utility vs noise (le={le}); ε annotated per point\n"
                   "green = reaches the ε budget", fontsize=9.5)

    # ---------------- 右：MAE vs ε ----------------
    for k in arms:
        maes = [float(r["final_mae"]) for r in sel[k]]
        eps = epsilon_of(sel[k][0])
        if eps is None or eps == float("inf"):
            continue
        mean, sd = statistics.mean(maes), (
            statistics.stdev(maes) if len(maes) > 1 else 0.0
        )
        good = eps <= eps_max
        ax_r.errorbar([eps], [mean], yerr=[sd],
                      color="C2" if good else "C0", marker="o", ms=8,
                      capsize=4, lw=1.4, zorder=4)
        for r in sel[k]:
            ax_r.plot([eps], [float(r["final_mae"])], marker="o", ms=3.2,
                      color="C2" if good else "C0", alpha=0.35, zorder=3)
        ax_r.annotate(f"nm={k[4]:g}", (eps, mean), textcoords="offset points",
                      xytext=(8, 6), fontsize=8.5,
                      color="C2" if good else "C0")
    tr_r = ax_r.get_xaxis_transform()
    if zero:
        maes = [float(r["final_mae"]) for r in sel[zero[0]]]
        ax_r.plot([0, 1], [statistics.mean(maes)] * 2, color="C1", ls="--",
                  lw=1.2, zorder=2, transform=tr_r,
                  label="nm=0 control (no guarantee)")
    if plain:
        maes = [float(r["final_mae"]) for r in sel[plain[0]]]
        ax_r.plot([0, 1], [statistics.mean(maes)] * 2, color="gray", ls=":",
                  lw=1.2, zorder=2, transform=tr_r,
                  label="no DP, same budget")
    # 同理不用 axvline(..., label=...)：它的 y 走轴分数、x 走数据，而横轴
    # 是对数的，图例同样会被排乱。get_yaxis_transform() 给出满高竖线。
    ax_r.plot([eps_max, eps_max], [0, 1], color="C3", ls="--", lw=1.2,
              zorder=2, transform=ax_r.get_yaxis_transform(),
              label=f"ε ≤ {eps_max:g} (acceptance)")
    # 判据真正允许的区域：ε ≤ 上限**且**劣化 ≤ 30%。右轴是 ε，所以这里只
    # 能画出 ε 那一半；纵向上那一半由非 DP 同预算基线 × 1.3 给出。两块
    # 交集为空 = 判据在这个协议下一个点都不满足，这比任何文字都直接。
    plain_mae = (statistics.mean(
        [float(r["final_mae"]) for r in sel[plain[0]]]) if plain else None)
    # 0 在对数轴上没有位置，用 1e-3 当左边界；不写死数据范围，交给后面的
    # set_xlim。写死 0 会让 autoscale 把横轴拉到 1e-14（2026-10-25 实测）。
    ax_r.axvspan(1e-3, eps_max, color="C3", alpha=0.07, zorder=0)
    if plain_mae:
        ax_r.axhspan(1e-6, plain_mae * 1.3, color="C2", alpha=0.07, zorder=0)
    ax_r.set_xscale("log")
    ax_r.set_yscale("log")
    ax_r.set_xlabel("ε at δ=1e-5 (smaller = more private)")
    ax_r.set_ylabel("test MAE")
    ax_r.grid(alpha=0.25, lw=0.5, which="both")
    # 横轴范围由数据定，不被 axvspan 的左边界 1e-3 拽走（否则 152 与 7.84
    # 之间会被压成一条线，而那正是全部结论所在的一段）。
    eps_seen = [epsilon_of(sel[k][0]) for k in arms]
    eps_seen = [e for e in eps_seen
                if e is not None and e != float("inf")]
    if eps_seen:
        lo, hi = min(eps_seen + [eps_max]), max(eps_seen + [eps_max])
        ax_r.set_xlim(lo / 2.5, hi * 2.5)
    ax_r.legend(fontsize=8, frameon=False, loc="upper left")
    ax_r.set_title("Privacy-utility trade-off; the shaded region is what the\n"
                   "acceptance criterion actually allows", fontsize=9.5)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"已写入 {out} 与 {out.with_suffix('.pdf')}")
    for k in arms:
        maes = [float(r["final_mae"]) for r in sel[k]]
        eps = epsilon_of(sel[k][0])
        eps_txt = "∞" if eps == float("inf") else f"{eps:.4g}"
        print(f"  nm={k[4]:<6g} C={k[5]:g}  ε={eps_txt:<10}  "
              f"MAE={statistics.mean(maes):.5f} ± {statistics.stdev(maes):.5f}  "
              f"(n={len(maes)})")
    for tag, k in (("nm=0 机制对照", zero[0] if zero else None),
                   ("非 DP 同预算", plain[0] if plain else None)):
        if k is None:
            continue
        maes = [float(r["final_mae"]) for r in sel[k]]
        print(f"  {tag}: MAE={statistics.mean(maes):.5f} (n={len(maes)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
