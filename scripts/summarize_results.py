#!/usr/bin/env python3
"""把 results/ 里的 run JSON 汇总成 plan.md §10.2 的表 1。

为什么要脚本而不是手抄：表 1 的每个数都要能追溯到具体文件，而且换 seed 数、
换 α 之后要重出。手抄一次之后，第二次就会开始"差不多就行"，那是造假的开始。

相对劣化的分母是 **C0 中心化**（plan.md §11.3）。C0 缺失时直接报错，
不拿 C1 顶替——分母换了而没记录，比没有分母更糟。

三种用法::

    # 人读（带 seed 波动检查、配对比较）
    python scripts/summarize_results.py --paired

    # 只出表格（markdown），用于贴进文档
    python scripts/summarize_results.py --format markdown

    # 把表格写回 plan.md 的生成区（CI 与本地都用它对齐）
    python scripts/summarize_results.py --write-plan docs/plan.md

    # 只核对，不写：表与 results/ 不一致就非零退出（CI 闸门）
    python scripts/summarize_results.py --check-plan docs/plan.md

    # 按 plan §11.3 的阈值判 H1/H2/H3
    python scripts/summarize_results.py --judge

**为什么要 --check-plan**：反造假闸门最初写的是「§10.2 的结果列必须全是
占位符 —」，那在有结果之前是对的。有了结果之后，那条断言只能拦住"提前
填数"，拦不住"填错的数"。改成逐字比对机器生成的表之后，文档里的每个
数字都必须能被 results/*.json 重新算出来——手改任何一个数都会红。

**为什么要 --judge**：判据原来靠人读「−20.9% 算不算超过 10%」，而 seed
波动 5%–63%，人读到哪一步都可能说服自己。改成机械判定：先做符号检验，
不显著一律「未判定」；显著了再按阈值分档。这样「未判定」是一个可复现的
结论，不是偷懒。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flsoh.stats import is_significant, paired_stats  # noqa: E402

BEGIN_MARK = "<!-- BEGIN GENERATED: results-table -->"
END_MARK = "<!-- END GENERATED: results-table -->"


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


def _key(run: dict) -> tuple:
    """分组键。

    用字段而不是 run_id 前缀：run_id 是人起的名字，改个后缀就会让分组
    静默错位，而字段是脚本自己写进 JSON 的。
    """
    return (
        str(run.get("strategy", "?")),
        run.get("scheme") or "—",
        run.get("alpha"),
        float(run.get("mu") or 0.0),
        run.get("local_epochs") if run.get("local_epochs") is not None
        else run.get("epochs"),
    )


def _label(key: tuple) -> str:
    """人读的配置名（不含编号）。"""
    strategy, scheme, alpha, mu, _ = key
    if strategy == "centralized":
        return "中心化"
    if strategy == "local":
        return "本地独立"
    name = "FedAvg" if strategy == "fedavg" else f"FedProx (μ={mu:g})"
    if scheme == "iid":
        return f"{name} / IID"
    if scheme == "dirichlet":
        return f"{name} / Dirichlet"
    return f"{name} / {scheme}"


# 编号与 plan.md §5 的假设表（H1→I1、H2→N1、H3→N1 vs N2）对应。
# 这些编号是人给的固定约定，不能由脚本随意生成——文档在引用它们。
_BASE_CODE = {
    ("centralized", "—"): "C0",
    ("local", "—"): "C1",
    ("fedavg", "iid"): "I1",
    ("fedavg", "dirichlet"): "N1",
    ("fedavg", "sohrank"): "P1",   # P = pathological，压力测试档
    ("fedprox", "dirichlet"): "N2",
}


def _codes(keys: list[tuple]) -> dict[tuple, str]:
    """给每组分配编号。

    同一编号出现多组时（α 扫描、μ 网格、不同 local_epochs 的等训练量对照）
    追加**只在碰撞组之间取值不同**的参数，否则表里会出现两个 N1 而读者
    分不清哪个对应哪一行。
    """
    raw = {}
    for key in keys:
        raw[key] = _BASE_CODE.get((key[0], key[1]), key[0][:2].upper())

    by_code: dict[str, list[tuple]] = {}
    for key, code in raw.items():
        by_code.setdefault(code, []).append(key)

    out = {}
    for code, group in by_code.items():
        if len(group) == 1:
            out[group[0]] = code
            continue
        # 只保留在组内真正有差异的字段，避免 N2@μ=0.1,le=10 这种噪声后缀
        varies_mu = len({k[3] for k in group}) > 1
        varies_alpha = len({k[2] for k in group}) > 1
        varies_le = len({k[4] for k in group}) > 1
        for key in group:
            parts = []
            if varies_mu:
                parts.append(f"μ={key[3]:g}")
            if varies_alpha and key[2] is not None:
                parts.append(f"α={key[2]:g}")
            if varies_le and key[4] is not None:
                parts.append(f"le={key[4]}")
            out[key] = f"{code}@{','.join(parts)}" if parts else code
    return out


def _group(runs: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        groups.setdefault(_key(r), []).append(r)
    return groups


def _by_seed(items: list[dict]) -> dict[int, float]:
    return {int(r.get("seed", -1)): r["final_mae"] for r in items}


def _paired(items_a: list[dict], items_b: list[dict]) -> dict | None:
    """按 seed 配对比两个配置。

    为什么需要配对：跨 seed 的绝对波动（本仓库实测 5%–63%）远大于要检测的
    效应（5%），所以单看两组的 mean±std 判不出差异。配对差把「seed 带来的
    共同波动」消掉，只留配置差异——这是当前数据下唯一有分辨力的做法。

    配对之后仍然要问「这个差显不显著」。**n=3 时符号检验的最小双侧 p 是
    0.25，任何效应都不可能显著**——所以「3 seeds 判不出来」是样本量问题，
    不是噪声问题。p 值由 flsoh.stats 算，不在这里拍脑袋。
    """
    a, b = _by_seed(items_a), _by_seed(items_b)
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None
    return paired_stats([a[s] - b[s] for s in common])


def _fmt_paired(tag_a: str, tag_b: str, res: dict, base_mae: float) -> str:
    rel = res["mean"] / base_mae * 100 if base_mae else float("nan")
    return (
        f"  {tag_a} − {tag_b}: 差 {res['mean']:+.5f} ± {res['sd']:.5f}  "
        f"(n={res['n']}, 相对 {rel:+.1f}%)  "
        f"符号 {res['n_neg']}/{res['n']} 为负  "
        f"p_sign={res['p_sign']:.4f}  p_t={res['p_t']:.4f}"
    )


def _heterogeneity(hetero_path: Path | None) -> dict:
    """读 results/heterogeneity.json（scripts/measure_heterogeneity.py 的输出）。

    文件缺失时返回空 dict，调用方按「无法做 manipulation check」处理，
    不静默当成「异质性没问题」。
    """
    if hetero_path is None or not Path(hetero_path).exists():
        return {}
    try:
        return json.loads(Path(hetero_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _skew_of(hetero: dict, scheme: str) -> float | None:
    """取某个划分方案造出的标签偏斜（客户端身份解释的方差占比）。"""
    for name, rep in (hetero.get("partitions") or {}).items():
        if f"-{scheme}" in name or name.endswith(f"{scheme}.json"):
            return rep.get("between_client_variance_fraction")
    return None


def judge_hypotheses(runs: list[dict], baseline_prefix: str = "c0__centralized",
                     hetero: dict | None = None) -> list[str]:
    """按 plan §11.3 的阈值机械判 H1/H2/H3。

    判定规则（先显著、再分档）：
    - 配对 t 的 p ≥ 0.05 → 一律「未判定」。不显著时「8% 还是 3%」的差别
      本身就在噪声里，按阈值划线等于在噪声里挑结论。主判据用 t 而不是
      符号检验，理由见 flsoh.stats.is_significant。
    - H1「IID 联邦与中心化的差距可以忽略」→ 判 |相对差|：
      ≤5% 接受、>10% 推翻。**用绝对值**是因为原判据写的是「差距」，
      联邦比中心化好 21% 同样是「差距不可忽略」。
    - H2「Non-IID 下 FedAvg 会显著掉点」→ 判带符号的相对劣化：
      >10% 接受、≤5% 推翻。
    - H3「FedProx 能挽回」→ 判相对 FedAvg 的改善量：≥5% 且 >1σ 接受。

    **判 H2 之前必须先做 manipulation check**：如果那份 Non-IID 划分造出的
    标签偏斜还不如 IID 基线（实测 Dirichlet α=0.5 是 12.2%，IID 是 17.3%），
    那「Non-IID 没掉点」是处理没施加，不是算法免疫——**这两种解释给出相反
    的结论，而 MAE 表上看起来一模一样**。这时只能判「未判定（处理未施加）」，
    否则等于拿一个没通电的电路去证明断路器好用。
    """
    hetero = hetero or {}
    groups = _group(runs)
    codes = _codes(list(groups))
    base = [r for r in runs if str(r.get("run_id", "")).startswith(baseline_prefix)]
    if not base:
        return ["  C0 缺失，无法判定"]
    base_key = _key(base[0])
    base_mae = statistics.mean(r["final_mae"] for r in base)
    out = []

    def rel(key) -> float:
        return (statistics.mean(r["final_mae"] for r in groups[key])
                - base_mae) / base_mae * 100

    def rel_to(key, other) -> float:
        """相对另一个配置组（不是 C0）的劣化。"""
        other_mae = statistics.mean(r["final_mae"] for r in groups[other])
        return (statistics.mean(r["final_mae"] for r in groups[key])
                - other_mae) / other_mae * 100

    # ---- H1：每个 IID FedAvg 组（不同 le 是不同口径，要分开判） ----
    out.append("H1（IID 联邦 vs 中心化，差距可以忽略？）")
    iid_keys = [k for k in groups if k[0] == "fedavg" and k[1] == "iid"]
    if not iid_keys:
        out.append("  没有 IID 的 FedAvg 结果")
    for k in sorted(iid_keys, key=lambda x: codes[x]):
        res = _paired(groups[k], groups[base_key])
        if res is None:
            out.append(f"  {codes[k]}: 共同 seed 不足 2 个")
            continue
        r = rel(k)
        sig, why = is_significant(res)
        if not sig:
            verdict = f"未判定（{why}）"
        elif abs(r) > 10:
            verdict = f"推翻（|{r:+.1f}%| > 10%，差距不可忽略；{why}）"
        elif abs(r) <= 5:
            verdict = f"接受（|{r:+.1f}%| ≤ 5%；{why}）"
        else:
            verdict = f"未判定（|{r:+.1f}%| 落在 5%–10%；{why}）"
        out.append(f"  {codes[k]}: 相对 C0 {r:+.1f}%  → {verdict}")

    # ---- H2：Non-IID FedAvg ----
    # 对照组是**同预算的 IID FedAvg**，不是 C0。理由：H2 问的是「划分方式
    # 造成多少损失」，联邦与中心化的差距是 H1 的问题。拿 C0 当分母会把两个
    # 效应混在一起——Dirichlet 相对 C0 好 35.9%，相对 IID 也好 16.3%，
    # 两个分母给出完全不同的画面。
    out.append("H2（Non-IID 下 FedAvg 显著掉点？对照组 = 同预算 IID FedAvg）")
    dir_keys = [k for k in groups if k[0] == "fedavg" and k[1] == "dirichlet"]
    # 偏斜分片也算 Non-IID：它用标签排序构造，是压力测试档
    rank_keys = [k for k in groups if k[0] == "fedavg" and k[1] == "sohrank"]
    if not dir_keys and not rank_keys:
        out.append("  没有 Non-IID 的 FedAvg 结果")

    for k in sorted(dir_keys + rank_keys, key=lambda x: codes[x]):
        scheme = k[1]
        # 同 local_epochs 的 IID 组才是对照组
        ctrl = ("fedavg", "iid", None, 0.0, k[4])
        if ctrl not in groups:
            out.append(f"  {codes[k]}: 缺同预算（local_epochs={k[4]}）的 IID 对照组")
            continue
        res = _paired(groups[k], groups[ctrl])
        if res is None:
            out.append(f"  {codes[k]}: 与 IID 的共同 seed 不足 2 个")
            continue
        r = rel_to(k, ctrl)
        skew = _skew_of(hetero, scheme)
        base_skew = _skew_of(hetero, "iid")
        if skew is None or base_skew is None:
            out.append(
                f"  {codes[k]}: 相对同预算 IID {r:+.1f}%  → "
                "未判定（缺 heterogeneity.json，无法做 manipulation check；"
                "先跑 scripts/measure_heterogeneity.py --out results/heterogeneity.json）"
            )
            continue
        if skew <= base_skew:
            out.append(
                f"  {codes[k]}: 相对同预算 IID {r:+.1f}%  → "
                f"未判定（处理未施加：该划分的标签偏斜 {skew * 100:.1f}% "
                f"≤ IID 基线的 {base_skew * 100:.1f}%。"
                "把 21 颗电池合成一个客户端会把电池间的 SOH 差异平均掉，"
                "所以这个划分实际上比 IID 更均匀）"
            )
            continue
        sig, why = is_significant(res)
        if not sig:
            verdict = f"未判定（{why}）"
        elif r > 10:
            verdict = f"接受（{r:+.1f}% > 10%，确实掉点；{why}）"
        elif r <= 5:
            verdict = f"推翻（{r:+.1f}% ≤ 5%，没有掉点；{why}）"
        else:
            verdict = f"未判定（{r:+.1f}% 落在 5%–10%；{why}）"
        out.append(f"  {codes[k]}: 偏斜 {skew * 100:.1f}%（IID {base_skew * 100:.1f}%）"
                   f"  → 相对同预算 IID {r:+.1f}%  → {verdict}")

    # ---- H3：FedProx vs 同划分 FedAvg ----
    out.append("H3（FedProx 能挽回损失？）")
    prox_keys = [k for k in groups if k[0] == "fedprox"]
    if not prox_keys:
        out.append("  没有 FedProx 结果")
    for k in sorted(prox_keys, key=lambda x: codes[x]):
        avg_key = ("fedavg", k[1], k[2], 0.0, k[4])
        if avg_key not in groups:
            out.append(f"  {codes[k]}: 缺同划分的 FedAvg")
            continue
        res = _paired(groups[k], groups[avg_key])
        if res is None:
            out.append(f"  {codes[k]}: 共同 seed 不足 2 个")
            continue
        avg_mae = statistics.mean(r["final_mae"] for r in groups[avg_key])
        improve = -(res["mean"] / avg_mae * 100)   # 正数 = FedProx 更好
        sig, why = is_significant(res)
        if not sig:
            verdict = f"未判定（{why}）"
        elif improve >= 5 and abs(res["mean"]) > res["sd"]:
            verdict = f"接受（改善 {improve:.1f}% ≥ 5%，且 |Δ| > 1σ；{why}）"
        else:
            verdict = (
                f"推翻（改善 {improve:+.1f}%，"
                f"{'未达 5%' if improve < 5 else '未超过 1σ'}；{why}）"
            )
        out.append(f"  {codes[k]} vs 同划分 FedAvg: 改善 {improve:+.1f}%  → {verdict}")

    return out


def build_table(runs: list[dict], baseline_prefix: str = "c0__centralized") -> list[str]:
    """生成表 1 的 markdown 行（含表头与分隔行）。"""
    base = [r for r in runs if str(r.get("run_id", "")).startswith(baseline_prefix)]
    if not base:
        raise SystemExit(
            f"找不到基线 {baseline_prefix}*。相对劣化必须有分母——"
            "先跑 scripts/run_local_baseline.py"
        )
    base_mae = statistics.mean(r["final_mae"] for r in base)

    lines = [
        "| 编号 | 配置 | 策略 | 客户端数 | 划分 | seeds | MAE (mean±std) | RMSE | 相对 C0 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    groups = _group(runs)
    codes = _codes(list(groups))
    for key, items in sorted(groups.items(), key=lambda kv: codes[kv[0]]):
        strategy, scheme, alpha, mu, _ = key
        maes = [r["final_mae"] for r in items]
        rmses = [r["final_rmse"] for r in items if r.get("final_rmse") is not None]
        n_clients = next((r.get("n_clients") for r in items
                          if r.get("n_clients") is not None), None)
        rel = (statistics.mean(maes) - base_mae) / base_mae * 100
        if key == _key(base[0]):
            rel_cell = "—"
        else:
            rel_cell = f"{rel:+.1f}%"
        split = "—" if scheme == "—" else scheme
        if alpha is not None:
            split = f"{split} α={alpha:g}"
        lines.append(
            f"| {codes[key]} | {_label(key)} | {strategy} "
            f"| {n_clients if n_clients is not None else '—'} "
            f"| {split} "
            f"| {len(items)} | {_fmt(maes)} | {_fmt(rmses)} | {rel_cell} |"
        )
    return lines


def _extract_block(text: str) -> str:
    if BEGIN_MARK not in text or END_MARK not in text:
        raise SystemExit(
            f"plan.md 里找不到生成区标记。表 1 必须夹在\n  {BEGIN_MARK}\n"
            f"  {END_MARK}\n之间，否则无法自动核对"
        )
    start = text.index(BEGIN_MARK) + len(BEGIN_MARK)
    end = text.index(END_MARK)
    return text[start:end].strip("\n")


def _replace_block(text: str, table: str) -> str:
    start = text.index(BEGIN_MARK) + len(BEGIN_MARK)
    end = text.index(END_MARK)
    return text[:start] + "\n" + table + "\n" + text[end:]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="汇总实验结果为表 1")
    ap.add_argument("--results", default="results")
    ap.add_argument("--baseline", default="c0__centralized",
                    help="相对劣化的分母，默认 C0 中心化")
    ap.add_argument("--format", choices=("text", "markdown"), default="text",
                    help="text：人读（含检查）；markdown：只出表")
    ap.add_argument("--paired", action="store_true",
                    help="额外输出按 seed 配对比较（噪声大时唯一有分辨力的做法）")
    ap.add_argument("--judge", action="store_true",
                    help="按 plan §11.3 的阈值机械判 H1/H2/H3")
    ap.add_argument("--hetero", default=None,
                    help="heterogeneity.json 路径；判 H2 前用它做 manipulation check")
    ap.add_argument("--write-plan", metavar="PLAN",
                    help="把表写回 plan.md 的生成区")
    ap.add_argument("--check-plan", metavar="PLAN",
                    help="只核对 plan.md 的生成区是否与 results/ 一致")
    args = ap.parse_args(argv)

    runs = _load(Path(args.results))
    if not runs:
        print(f"FAIL {args.results} 里没有带 final_mae 的 run")
        return 1

    table = "\n".join(build_table(runs, args.baseline))

    if args.check_plan:
        path = Path(args.check_plan)
        current = _extract_block(path.read_text(encoding="utf-8"))
        if current != table:
            print("FAIL §10.2 的表与 results/ 不一致。")
            print("  期望（由 results/*.json 重新算出）：")
            for line in table.splitlines():
                print("    " + line)
            print("  实际（文档里写的）：")
            for line in current.splitlines():
                print("    " + line)
            print("  修法：python scripts/summarize_results.py --write-plan "
                  f"{args.check_plan}")
            return 3
        print(f"OK §10.2 的表与 results/ 逐字一致（{len(runs)} 个 run）")
        return 0

    if args.write_plan:
        path = Path(args.write_plan)
        text = path.read_text(encoding="utf-8")
        _extract_block(text)  # 先确认标记存在，避免写坏
        path.write_text(_replace_block(text, table), encoding="utf-8")
        print(f"已写入 {path}")
        return 0

    if args.format == "markdown":
        print(table)
        return 0

    base = [r for r in runs if str(r.get("run_id", "")).startswith(args.baseline)]
    groups = _group(runs)
    codes = _codes(list(groups))
    tag = lambda k: f"{codes[k]} {_label(k)}"  # noqa: E731
    print(f"基线 C0 中心化：MAE "
          f"{statistics.mean(r['final_mae'] for r in base):.5f}"
          f"（{len(base)} 个 seed）\n")
    print(table)

    # seed 波动：plan.md §11.3 要求相对标准差 <10%，超了要加 seed 数
    print("\nseed 波动检查（相对标准差，阈值 10%）：")
    bad = []
    for key, items in sorted(groups.items(), key=lambda kv: codes[kv[0]]):
        maes = [r["final_mae"] for r in items]
        if len(maes) < 2:
            print(f"  {tag(key)}: 只有 {len(maes)} 个 seed，无法评估波动")
            continue
        mean = statistics.mean(maes)
        cv = statistics.stdev(maes) / mean * 100 if mean else float("inf")
        flag = "OK" if cv < 10 else "**超标**"
        if cv >= 10:
            bad.append((tag(key), cv))
        print(f"  {tag(key)}: {cv:.1f}%  {flag}")

    if bad:
        print("\n超标项需要加 seed 数，不许挑最好的一次报（plan.md §11.3）。")

    if args.paired:
        print("\n配对比较（同 seed 相减，消掉 seed 共同波动）：")
        base_key = _key(base[0])
        base_mae = statistics.mean(r["final_mae"] for r in base)
        for key, items in sorted(groups.items(), key=lambda kv: codes[kv[0]]):
            if key == base_key:
                continue
            res = _paired(items, groups[base_key])
            if res is None:
                print(f"  {tag(key)} − C0: 共同 seed 不足 2 个，跳过")
                continue
            print(_fmt_paired(tag(key), "C0", res, base_mae))
        # FedProx 的对照是 FedAvg，不是 C0（§11.3：H3 是「比 FedAvg 改善」）
        for key, items in sorted(groups.items(), key=lambda kv: codes[kv[0]]):
            if key[0] != "fedprox":
                continue
            avg_key = ("fedavg", key[1], key[2], 0.0, key[4])
            if avg_key not in groups:
                print(f"  {tag(key)} − FedAvg: 缺同划分的 FedAvg，跳过")
                continue
            res = _paired(items, groups[avg_key])
            if res is None:
                print(f"  {tag(key)} − FedAvg: 共同 seed 不足 2 个，跳过")
                continue
            avg_mae = statistics.mean(r["final_mae"] for r in groups[avg_key])
            print(_fmt_paired(tag(key), f"{codes[avg_key]} 同划分", res, avg_mae))

    if args.judge:
        print("\n假设判定（plan §11.3：先配对 t 检验，不显著一律未判定；"
              "判 H2 前先做 manipulation check）：")
        for line in judge_hypotheses(
            runs, args.baseline, _heterogeneity(Path(args.hetero))
        ):
            print("  " + line if not line.startswith("H") else "\n" + line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
