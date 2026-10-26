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


def _tail_mae(run: dict, points: int) -> float | None:
    """末 ``points`` 个记录点的平均 MAE；曲线不够长返回 None。

    为什么需要这个口径：`final_mae` 是**轨迹上的一个采样点**，而轨迹本身
    是尖峰的。C0 的 8 个 seed 实测（同数据、同 1 线程、可逐位复现）：

    | 口径 | 均值 | CV |
    |---|---|---|
    | 第 300 epoch 瞬时值 | 0.002709 | **77.2%** |
    | 末 30 epoch 中位 | 0.002105 | **23.7%** |

    77.2% 这个数一度让 H1/H2/H3 全部退回「未判定」——但其中三分之二的
    波动来自「停在哪个相位」，不是「学到了什么」。末段平均把这一部分消掉。
    """
    maes = [float(p["mae"]) for p in (run.get("curve") or []) if "mae" in p]
    if len(maes) < points:
        return None
    return statistics.mean(maes[-points:])


def _apply_metric(runs: list[dict], metric: str, points: int) -> tuple[int, int]:
    """把选定口径写进 ``final_mae``，原值留在 ``last_round_mae``。

    下游 20 多处都读 ``final_mae``，就地归一化比逐个改调用点安全——少改
    一处就会得到「一半末段平均、一半末轮」的混合表，而这种表看起来完全
    正常。返回 ``(改写的 run 数, 没有曲线而回退的 run 数)``。
    """
    if metric == "final":
        return 0, 0
    changed = fell_back = 0
    for run in runs:
        if "last_round_mae" not in run:
            run["last_round_mae"] = run["final_mae"]
        tail = _tail_mae(run, points)
        if tail is None:
            fell_back += 1
            continue
        run["final_mae"] = tail
        changed += 1
    return changed, fell_back


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


def _server_key(run: dict) -> tuple | None:
    """服务端优化器配置；只有 FedAdam 有，其余策略返回 None。

    返回整个四元组而不是只取 η：2026-10-24 真的发生过——η=0.01 的正式
    实验与 η=0.01、τ=1e-5 的探针**只差 τ**，两组塌成一行后
    ``_by_seed`` 用后一个文件静默覆盖了 seed 0（0.00101 → 0.00128），
    配对差从 +0.00002 变成 +0.00006，而 seeds 列还写着 8。
    一个维度没进键，表现不是「少一行」，是**某一行的数被换掉**。
    """
    if str(run.get("strategy", "?")) != "fedadam":
        return None
    return (
        run.get("server_lr"),
        run.get("server_beta_1"),
        run.get("server_beta_2"),
        run.get("server_tau"),
    )


def _fmt_server(server: tuple | None) -> str:
    """配置列里的服务端超参片段。"""
    if server is None:
        return ""
    eta, beta_1, beta_2, tau = server
    parts = []
    if eta is not None:
        parts.append(f"η={eta:g}")
    if tau is not None:
        parts.append(f"τ={tau:g}")
    if beta_1 is not None or beta_2 is not None:
        parts.append(f"β=({beta_1:g},{beta_2:g})")
    return f" ({', '.join(parts)})" if parts else ""


def _dp_key(run: dict) -> tuple | None:
    """本地 DP 配置；没开 DP 的 run 返回 None。

    返回三元组 ``(noise_multiplier, max_grad_norm, ε)`` 而不是只取
    ``noise_multiplier``——理由与 :func:`_server_key` 完全相同，而且这次
    连"只取主超参"都不够：**ε 也进键**。ε 由 (nm, 抽样率, 步数) 唯一决定，
    所以它进键不会把本该同组的 run 拆开；但它进键之后，标签才能显示
    「每点附 ε」（M4 的 DOD ①），而不必让 :func:`_label` 去翻 run 列表。

    ``noise_multiplier=0`` 的对照组与正式臂只差 nm，必须分开——它们测的
    是两件事：前者是「DP 机制（Poisson + 裁剪）的开销」，后者是「噪声的
    开销」。
    """
    if run.get("noise_multiplier") is None:
        return None
    eps = run.get("dp_epsilon")
    eps = float("inf") if eps is None else float(eps)
    # 四舍五入到 0.01：ε 是同一个确定函数的输出，同组内逐位相同；
    # 取整只为让标签短一点，不为合并任何东西。
    return (
        float(run["noise_multiplier"]),
        float(run.get("max_grad_norm") or 0.0),
        round(eps, 2),
    )


def _fmt_dp(dp: tuple | None) -> str:
    """配置列里的隐私超参片段。"""
    if dp is None:
        return ""
    nm, norm, eps = dp
    if nm == 0.0:
        # 对照组：没有噪声就没有 DP（plan §7）。写成"无保证"而不是 ε=∞，
        # 因为 ∞ 在表里会被读成"数值很大"，而它的含义是"这一行不是 DP"。
        return f" + DP 机制 (nm=0, C={norm:g}, 无保证)"
    eps_txt = "∞" if eps == float("inf") else f"{eps:g}"
    return f" + DP (nm={nm:g}, C={norm:g}, ε={eps_txt})"


def _key(run: dict) -> tuple:
    """分组键。

    用字段而不是 run_id 前缀：run_id 是人起的名字，改个后缀就会让分组
    静默错位，而字段是脚本自己写进 JSON 的。

    **`n_clients` 必须在键里**（M3 的 k 扫描，2026-10-24）。sohrank 的
    k=1/2/5/10 四档共用同一个 (strategy, scheme, local_epochs)，实测 MAE
    却是 0.0038 / 0.0038 / 0.0015 / 0.0014——差 2.7 倍。少了这一维，四档
    会塌成一行，均值描述的是任何一个都不存在的实验，而表看起来照样正常。
    同一维度还决定「每客户端拿几颗电池」，是 k 扫描要控制的混杂量本身。

    **服务端配置也必须在键里**（M3 ② 的 FedAdam，2026-10-24）。η/τ 探针
    与正式实验只差一个 τ，实测 MAE 差 0.00101 / 0.00128 / 0.00173。
    见 :func:`_server_key` 的 docstring——那一维漏掉时被静默换掉的是
    某一行的数，不是行数。

    **隐私配置同理**（M4 的 DP，2026-10-25）。噪声扫描五档只差一个 nm，
    而 nm=1 与 nm=8 的 MAE 预期差一个量级。见 :func:`_dp_key`。
    """
    return (
        str(run.get("strategy", "?")),
        run.get("scheme") or "—",
        run.get("alpha"),
        float(run.get("mu") or 0.0),
        run.get("local_epochs") if run.get("local_epochs") is not None
        else run.get("epochs"),
        run.get("n_clients"),
        _server_key(run),
        _dp_key(run),
    )


def _label(key: tuple) -> str:
    """人读的配置名（不含编号）。"""
    strategy, scheme, alpha, mu, _, _, server, dp = key
    if strategy == "centralized":
        return "中心化"
    if strategy == "local":
        return "本地独立"
    if strategy == "fedadam":
        name = f"FedAdam{_fmt_server(server)}"
    else:
        name = "FedAvg" if strategy == "fedavg" else f"FedProx (μ={mu:g})"
    if scheme == "iid":
        out = f"{name} / IID"
    elif scheme == "dirichlet":
        out = f"{name} / Dirichlet"
    else:
        out = f"{name} / {scheme}"
    return out + _fmt_dp(dp)


# 编号与 plan.md §5 的假设表（H1→I1、H2→N1、H3→N1 vs N2）对应。
# 这些编号是人给的固定约定，不能由脚本随意生成——文档在引用它们。
_BASE_CODE = {
    ("centralized", "—"): "C0",
    ("local", "—"): "C1",
    ("fedavg", "iid"): "I1",
    ("fedavg", "dirichlet"): "N1",
    ("fedavg", "sohrank"): "P1",   # P = pathological，压力测试档
    ("fedprox", "dirichlet"): "N2",
    ("fedadam", "iid"): "A1",      # A = Adam，服务端优化器组（M3 ②）
}


def _codes(keys: list[tuple]) -> dict[tuple, str]:
    """给每组分配编号。

    同一编号出现多组时（α 扫描、μ 网格、不同 local_epochs 的等训练量对照、
    M3 的 k 扫描、M4 的噪声扫描）追加**只在碰撞组之间取值不同**的参数，
    否则表里会出现两个 N1 而读者分不清哪个对应哪一行。

    DP 组单独拿一个基编号 ``D1``（D = differential privacy）：它与 I1 的
    strategy/scheme 完全相同，只差客户端侧加没加噪。若沿用 I1，五档噪声会
    变成 ``I1@le=3,nm=1`` 这样的后缀，读者要数五个后缀才知道自己在看
    隐私实验；而它确实**不是**同一族实验——分母都变了。
    """
    raw = {}
    for key in keys:
        if key[7] is not None:
            raw[key] = "D1"
        else:
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
        varies_n = len({k[5] for k in group}) > 1
        varies_server = len({k[6] for k in group}) > 1
        varies_dp = len({k[7] for k in group}) > 1
        for key in group:
            parts = []
            if varies_n and key[5] is not None:
                parts.append(f"N={key[5]}")
            if varies_mu:
                parts.append(f"μ={key[3]:g}")
            if varies_alpha and key[2] is not None:
                parts.append(f"α={key[2]:g}")
            if varies_le and key[4] is not None:
                parts.append(f"le={key[4]}")
            if varies_server:
                # 组内只标真正不同的那一项：η 探针三档只差 η，τ 探针只差 τ
                etas = {k[6][0] for k in group if k[6]}
                taus = {k[6][3] for k in group if k[6]}
                if len(etas) > 1:
                    parts.append(f"η={key[6][0]:g}")
                if len(taus) > 1:
                    parts.append(f"τ={key[6][3]:g}")
            if varies_dp:
                # ε 由 nm 唯一决定，标 nm 就够；ε 在配置列里
                parts.append(f"nm={key[7][0]:g}")
            out[key] = f"{code}@{','.join(parts)}" if parts else code
    return out


def _sort_key(key: tuple, code: str):
    """组的排序键：先按基编号，再按 N、μ、α、le、服务端 η/τ、隐私 nm 数值排。

    不能直接按 code 字符串排：`"I1@N=115" < "I1@N=12"`（字典序），表里
    115 会跑到 12 前面去。所以拆成 (基编号, 数值字段) 元组。
    """
    server = key[6] or (None, None, None, None)
    dp = key[7] or (None, None, None)
    return (
        code.split("@")[0],
        key[5] if key[5] is not None else 0,
        key[3],
        key[2] if key[2] is not None else -1.0,
        key[4] if key[4] is not None else -1,
        server[0] if server[0] is not None else -1.0,
        server[3] if server[3] is not None else -1.0,
        dp[0] if dp[0] is not None else -1.0,
    )


def _sorted_groups(groups: dict, codes: dict) -> list[tuple]:
    return sorted(groups.items(), key=lambda kv: _sort_key(kv[0], codes[kv[0]]))


def _group(runs: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        groups.setdefault(_key(r), []).append(r)
    return groups


def _by_seed(items: list[dict]) -> dict[int, float]:
    return {int(r.get("seed", -1)): r["final_mae"] for r in items}


def seed_cv_report(
    groups: dict, codes: dict, tag
) -> tuple[list[str], list[tuple[str, float]]]:
    """每个配置的 seed 相对标准差（CV）。返回（要打印的行，超标项）。"""
    lines: list[str] = []
    bad: list[tuple[str, float]] = []
    for key, items in _sorted_groups(groups, codes):
        maes = [r["final_mae"] for r in items]
        if len(maes) < 2:
            lines.append(f"  {tag(key)}: 只有 {len(maes)} 个 seed，无法评估波动")
            continue
        mean = statistics.mean(maes)
        cv = statistics.stdev(maes) / mean * 100 if mean else float("inf")
        lines.append(f"  {tag(key)}: {cv:.1f}%  {'OK' if cv < 10 else '**超标**'}")
        if cv >= 10:
            bad.append((tag(key), cv))
    return lines, bad


def seed_cv_advice(bad: list[tuple[str, float]]) -> str | None:
    """CV 超标时该怎么读——**不是**「回去加 seed」。

    这条提示上一版写的是「超标项需要加 seed 数」。那句话单独看没错（plan
    §11.3 的判据表就是这么写的），但它与 plan §11.3.1 的结论矛盾：那里已经
    算过，按 CV=16%、想分辨 5% 要 ~50 个 seed，4 配置 × 50 seed = 10–20
    小时，代价与收益不成比例，所以最终选的是**路径 3（配对比较）**。

    于是同一段输出会自相矛盾：上一句说「你必须加 seed」，下一段就打印配对
    比较的 p 值。读者按前一句理解，会把一份已经完成的判定当成待办事项。
    """
    if not bad:
        return None
    worst_tag, worst_cv = max(bad, key=lambda kv: kv[1])
    return (
        f"\n{len(bad)} 个配置的 seed 波动超过 10%（最差 {worst_tag} {worst_cv:.1f}%）。"
        "这不是「再抽几个 seed」能修的：CV 16% 想分辨 5% 要 ~50 个 seed，"
        "4 配置 × 50 seed = 10–20 小时，代价与收益不成比例（plan.md §11.3.1）。\n"
        "  它的含义是**不能拿绝对 MAE 跨配置比**——噪声比要检测的效应还大。"
        "判据改用下面的配对比较（§11.3.2）：同 seed 相减，seed 的共同波动被"
        "消掉，只留配置差异。只有配对差本身仍小于 seed 噪声时才真需要加 seed。"
    )


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


def _skew_of(hetero: dict, scheme: str, n_clients: int | None = None) -> float | None:
    """取某个划分方案造出的标签偏斜（客户端身份解释的方差占比）。

    `n_clients` 必须一起给：M3 的 k 扫描让同一 scheme 对应四份划分
    （N=12/23/58/115），偏斜从 4.7% 到 70.4%。不给 N 就会永远返回文件里
    第一份匹配项——k=10 的实验拿到 k=1 的偏斜，manipulation check 随之
    失效，而输出看起来完全正常。
    """
    for name, rep in (hetero.get("partitions") or {}).items():
        if not (f"-{scheme}" in name or name.endswith(f"{scheme}.json")):
            continue
        if n_clients is not None and rep.get("n_clients") != n_clients:
            continue
        return rep.get("between_client_variance_fraction")
    return None


def judge_hypotheses(runs: list[dict], baseline_prefix: str = "c0__centralized",
                     hetero: dict | None = None) -> list[str]:
    """按 plan §11.3 的阈值机械判 H1/H2/H3，以及 M4 的 H4。

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

    # ---- H1：每个 IID FedAvg 组（不同 le 是不同口径、不同 N 是不同规模，
    #      都要分开判；MVP 口径是 le=10、N=23） ----
    out.append("H1（IID 联邦 vs 中心化，差距可以忽略？）")
    # DP 臂（k[7] is not None）不在这里判：它们也在 IID 上，但 H1 问的是
    # **没有隐私机制时**联邦能不能追平。把五档 DP 也列进 H1，同一个 MAE 表
    # 就会在两个假设下各出现一次，而读者无法从行上看出来它们是另一种处理。
    # DP 的结论统一在下面的 M4 / H4 段，那里有 nm=0 机制对照做分母。
    iid_keys = [k for k in groups
                if k[0] == "fedavg" and k[1] == "iid" and k[7] is None]
    if not iid_keys:
        out.append("  没有 IID 的 FedAvg 结果")
    for k in sorted(iid_keys, key=lambda x: _sort_key(x, codes[x])):
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
    dir_keys = [k for k in groups
                if k[0] == "fedavg" and k[1] == "dirichlet" and k[7] is None]
    # 偏斜分片也算 Non-IID：它用标签排序构造，是压力测试档
    rank_keys = [k for k in groups
                 if k[0] == "fedavg" and k[1] == "sohrank" and k[7] is None]
    if not dir_keys and not rank_keys:
        out.append("  没有 Non-IID 的 FedAvg 结果")

    for k in sorted(dir_keys + rank_keys, key=lambda x: _sort_key(x, codes[x])):
        scheme = k[1]
        # 同 local_epochs、同 N（同 k）的 IID 组才是对照组。
        # N 也必须相同：k 扫描里「客户端更多」本身就会掉点（每客户端数据更少、
        # 梯度步更少），拿不同 N 的 IID 当分母会把 N 的效应算到偏斜头上。
        ctrl = ("fedavg", "iid", None, 0.0, k[4], k[5], None, None)
        if ctrl not in groups:
            out.append(f"  {codes[k]}: 缺同预算（local_epochs={k[4]}、N={k[5]}）的 IID 对照组")
            continue
        res = _paired(groups[k], groups[ctrl])
        if res is None:
            out.append(f"  {codes[k]}: 与 IID 的共同 seed 不足 2 个")
            continue
        r = rel_to(k, ctrl)
        skew = _skew_of(hetero, scheme, k[5])
        base_skew = _skew_of(hetero, "iid", k[5])
        if skew is None or base_skew is None:
            out.append(
                f"  {codes[k]}: 相对同预算 IID {r:+.1f}%  → "
                "未判定（heterogeneity.json 里找不到该 N 的划分，无法做 "
                "manipulation check；先跑 scripts/measure_heterogeneity.py "
                "--out results/heterogeneity.json）"
            )
            continue
        if skew < base_skew:
            out.append(
                f"  {codes[k]}: 相对同预算 IID {r:+.1f}%  → "
                f"未判定（处理未施加：该划分的标签偏斜 {skew * 100:.1f}% "
                f"< IID 同 N 基线的 {base_skew * 100:.1f}%。"
                "把多颗电池合成一个客户端会把电池间的 SOH 差异平均掉，"
                "所以这个划分实际上比 IID 更均匀）"
            )
            continue
        if skew == base_skew:
            out.append(
                f"  {codes[k]}: 相对同预算 IID {r:+.1f}%  → "
                f"未判定（处理未施加：该划分与 IID 同 N 划分的标签偏斜"
                f"完全相同（{skew * 100:.1f}%）。一颗电池一个客户端时，"
                "排序只改客户端的标签、不改客户端均值的分布，划分在统计上"
                "与 IID 等价——这一档是图 2 的内部校准点，不是处理组）"
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
        out.append(f"  {codes[k]}: 偏斜 {skew * 100:.1f}%（IID 同 N {base_skew * 100:.1f}%）"
                   f"  → 相对同预算 IID {r:+.1f}%  → {verdict}")

    # ---- H3：FedProx vs 同划分 FedAvg ----
    out.append("H3（FedProx 能挽回损失？）")
    prox_keys = [k for k in groups if k[0] == "fedprox"]
    if not prox_keys:
        out.append("  没有 FedProx 结果")
    for k in sorted(prox_keys, key=lambda x: _sort_key(x, codes[x])):
        avg_key = ("fedavg", k[1], k[2], 0.0, k[4], k[5], None, None)
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

    # ---- M3 ②：服务端 Adam 能否替代客户端的训练量（对 H3 机理的追问） ----
    # 这不是新假设，是对「每轮聚合后客户端动量归零」这条机理的直接检验：
    # 如果动量归零是绑定约束，FedAdam 应当用更少的本地 epoch 追平 FedAvg。
    # 对照组必须是**同 local_epochs、同 N 的 FedAvg**——换任何一个都会
    # 把训练量的差异算到服务端优化器头上（§14 坑 15：不声明训练量的
    # 相对差会反号）。
    out.append("M3 ②（FedAdam vs 同预算 FedAvg：服务端动量能否替代训练量？）")
    adam_keys = [k for k in groups if k[0] == "fedadam"]
    if not adam_keys:
        out.append("  没有 FedAdam 结果")
    for k in sorted(adam_keys, key=lambda x: _sort_key(x, codes[x])):
        ctrl = ("fedavg", k[1], None, 0.0, k[4], k[5], None, None)
        if ctrl not in groups:
            out.append(
                f"  {codes[k]}: 缺同预算（local_epochs={k[4]}、N={k[5]}）的 FedAvg 对照组"
            )
            continue
        res = _paired(groups[k], groups[ctrl])
        if res is None:
            out.append(f"  {codes[k]}: 共同 seed 不足 2 个")
            continue
        avg_mae = statistics.mean(r["final_mae"] for r in groups[ctrl])
        improve = -(res["mean"] / avg_mae * 100)   # 正数 = FedAdam 更好
        sig, why = is_significant(res)
        if not sig:
            verdict = f"未判定（{why}）"
        elif improve >= 5:
            verdict = f"FedAdam 更好（{improve:+.1f}%；{why}）"
        elif improve <= -5:
            verdict = f"FedAdam 更差（{improve:+.1f}%；{why}）"
        else:
            verdict = f"未判定（{improve:+.1f}% 落在 ±5% 内；{why}）"
        out.append(
            f"  {codes[k]} vs {codes[ctrl]} 同预算: 改善 {improve:+.1f}%  → {verdict}"
        )

    # ---- M4 / H4：本地 DP 的精度代价随噪声单调上升？ε ≤ 8 时仍可用？ ----
    # H4 有两半，必须分开判，因为它们的对照组不同：
    #   (a) 「代价随噪声单调上升」—— 分母是 nm=0 的机制对照组
    #       （同 Poisson、同裁剪、只差不加噪）。没有它，nm→0 的外推点与
    #       非 DP 基线之间的落差会被误读成"小噪声很便宜"。
    #   (b) 「ε ≤ 8 时仍可用」—— 这是 plan §11.3 的 DP 可用性判据
    #       （劣化 ≤30%）。ε 由 (nm, 抽样率, 步数) 决定，不是自由变量，
    #       所以「ε ≤ 8」等价于「nm 大到某个值」，而 nm 越大噪声越狠——
    #       这两半在同一端相遇，正是 H4 的张力所在。
    out.append("M4 / H4（本地 DP：精度代价随噪声单调上升？ε ≤ 8 时仍可用？）")
    dp_keys = [k for k in groups if k[7] is not None]
    if not dp_keys:
        out.append("  没有 DP 结果")
        return out
    # 按 (scheme, local_epochs, N) 分族：不同训练预算的 DP 不可比
    families: dict[tuple, list[tuple]] = {}
    for k in dp_keys:
        families.setdefault((k[1], k[4], k[5]), []).append(k)
    for fam in sorted(families, key=lambda f: (f[0], f[1] or -1, f[2] or 0)):
        scheme, le, n_clients = fam
        keys = sorted(families[fam], key=lambda x: _sort_key(x, codes[x]))
        # 非 DP 的同预算基线：用来回答"DP 一共付出多少"
        plain = ("fedavg", scheme, None, 0.0, le, n_clients, None, None)
        plain_mae = (
            statistics.mean(r["final_mae"] for r in groups[plain])
            if plain in groups else None
        )
        # nm=0 的机制对照组：用来回答"噪声单独付出多少"
        zero_keys = [k for k in keys if k[7][0] == 0.0]
        zero_mae = (
            statistics.mean(r["final_mae"] for r in groups[zero_keys[0]])
            if zero_keys else None
        )
        head = f"  [{scheme} le={le} N={n_clients}]"
        if plain_mae is not None:
            head += f" 非 DP 同预算 {plain_mae:.5f}"
        if zero_mae is not None:
            head += f"；nm=0 机制对照 {zero_mae:.5f}"
        else:
            head += "；**缺 nm=0 对照组，噪声与机制的开销无法分离**"
        out.append(head)

        arms = [k for k in keys if k[7][0] > 0]
        if not arms:
            out.append("    只有对照组，没有加噪臂")
            continue
        maes = []
        for k in arms:
            nm = k[7][0]
            eps = k[7][2]
            mae = statistics.mean(r["final_mae"] for r in groups[k])
            maes.append(mae)
            bits = [f"nm={nm:g}", f"MAE={mae:.5f}"]
            if eps != float("inf"):
                bits.append(f"ε={eps:g}")
            if zero_mae:
                bits.append(f"相对机制对照 {(mae - zero_mae) / zero_mae * 100:+.1f}%")
            if plain_mae:
                bits.append(f"相对非 DP {(mae - plain_mae) / plain_mae * 100:+.1f}")
            out.append("    " + "  ".join(bits))
            if zero_mae and k[7][0] > 0:
                res = _paired(groups[k], groups[zero_keys[0]])
                if res is not None:
                    sig, why = is_significant(res)
                    out.append(
                        f"      配对 vs nm=0: {res['mean']:+.5f} ± {res['sd']:.5f} "
                        f"(n={res['n']}, p_t={res['p_t']:.4f}, p_sign={res['p_sign']:.4f})"
                        f"  {'显著' if sig else '不显著：' + why}"
                    )

        # (a) 单调性：MAE 必须随 nm 不减
        if len(maes) >= 2:
            drops = [
                (arms[i][7][0], arms[i + 1][7][0])
                for i in range(len(maes) - 1)
                if maes[i + 1] < maes[i]
            ]
            if drops:
                pairs = "、".join(f"nm={a:g}→{b:g}" for a, b in drops)
                out.append(f"    → 单调性**不成立**（{pairs} 处 MAE 下降）")
            else:
                out.append("    → 单调性成立（MAE 随 nm 不减）")
        # (b) ε ≤ 8 且劣化 ≤30%（plan §11.3 的 DP 可用性判据）
        usable = [
            k for k in arms
            if k[7][2] != float("inf") and k[7][2] <= 8.0
        ]
        if not usable:
            out.append(
                "    → ε ≤ 8 的一档不存在：当前步数 × 抽样率下，nm 再大也只能"
                "把 ε 压到 8 以上（见 plan §13.4 的测算表）"
            )
        else:
            for k in usable:
                mae = statistics.mean(r["final_mae"] for r in groups[k])
                ref = zero_mae if zero_mae else plain_mae
                if ref is None:
                    out.append(f"    → ε={k[7][2]:g} 缺对照组，无法判劣化")
                    continue
                deg = (mae - ref) / ref * 100
                verdict = "可用" if deg <= 30 else "代价过高"
                out.append(
                    f"    → ε={k[7][2]:g}（nm={k[7][0]:g}）劣化 {deg:+.1f}% "
                    f"→ {verdict}（阈值 30%）"
                )
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
    for key, items in _sorted_groups(groups, codes):
        strategy, scheme, alpha, mu, _, n_clients, _, _ = key
        maes = [r["final_mae"] for r in items]
        rmses = [r["final_rmse"] for r in items if r.get("final_rmse") is not None]
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
                    help="按 plan §11.3 的阈值机械判 H1/H2/H3/M4-H4")
    ap.add_argument("--hetero", default=None,
                    help="heterogeneity.json 路径；判 H2 前用它做 manipulation check")
    ap.add_argument("--write-plan", metavar="PLAN",
                    help="把表写回 plan.md 的生成区")
    ap.add_argument("--check-plan", metavar="PLAN",
                    help="只核对 plan.md 的生成区是否与 results/ 一致")
    ap.add_argument("--metric", choices=("final", "tail"), default="final",
                    help="final=末轮瞬时值（历史口径）；tail=末 N 个记录点的平均")
    ap.add_argument("--tail-points", type=int, default=10,
                    help="--metric tail 的窗口（默认 10 个记录点）")
    args = ap.parse_args(argv)

    runs = _load(Path(args.results))
    if not runs:
        print(f"FAIL {args.results} 里没有带 final_mae 的 run")
        return 1

    changed, fell_back = _apply_metric(runs, args.metric, args.tail_points)
    if args.metric != "final":
        print(f"口径 = 末 {args.tail_points} 点平均：改写 {changed} 个 run"
              + (f"，{fell_back} 个没有足够曲线、回退到末轮值" if fell_back else ""),
              file=sys.stderr)

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

    # seed 波动：plan.md §11.3 要求相对标准差 <10%
    print("\nseed 波动检查（相对标准差，阈值 10%）：")
    cv_lines, bad = seed_cv_report(groups, codes, tag)
    for line in cv_lines:
        print(line)
    advice = seed_cv_advice(bad)
    if advice:
        print(advice)

    if args.paired:
        print("\n配对比较（同 seed 相减，消掉 seed 共同波动）：")
        base_key = _key(base[0])
        base_mae = statistics.mean(r["final_mae"] for r in base)
        for key, items in _sorted_groups(groups, codes):
            if key == base_key:
                continue
            res = _paired(items, groups[base_key])
            if res is None:
                print(f"  {tag(key)} − C0: 共同 seed 不足 2 个，跳过")
                continue
            print(_fmt_paired(tag(key), "C0", res, base_mae))
        # FedProx 的对照是 FedAvg，不是 C0（§11.3：H3 是「比 FedAvg 改善」）
        for key, items in _sorted_groups(groups, codes):
            if key[0] != "fedprox":
                continue
            avg_key = ("fedavg", key[1], key[2], 0.0, key[4], key[5], None, None)
            if avg_key not in groups:
                print(f"  {tag(key)} − FedAvg: 缺同划分的 FedAvg，跳过")
                continue
            res = _paired(items, groups[avg_key])
            if res is None:
                print(f"  {tag(key)} − FedAvg: 共同 seed 不足 2 个，跳过")
                continue
            avg_mae = statistics.mean(r["final_mae"] for r in groups[avg_key])
            print(_fmt_paired(tag(key), f"{codes[avg_key]} 同划分", res, avg_mae))
        # FedAdam 的对照也是 FedAvg，但必须是**同预算**（同 le、同 N）：
        # 换 le 会把训练量的差异算到服务端优化器头上（§14 坑 15）。
        for key, items in _sorted_groups(groups, codes):
            if key[0] != "fedadam":
                continue
            avg_key = ("fedavg", key[1], None, 0.0, key[4], key[5], None, None)
            if avg_key not in groups:
                print(f"  {tag(key)} − FedAvg: 缺同预算的 FedAvg，跳过")
                continue
            res = _paired(items, groups[avg_key])
            if res is None:
                print(f"  {tag(key)} − FedAvg: 共同 seed 不足 2 个，跳过")
                continue
            avg_mae = statistics.mean(r["final_mae"] for r in groups[avg_key])
            print(_fmt_paired(tag(key), f"{codes[avg_key]} 同预算", res, avg_mae))
        # DP 的对照有两层，都要报：
        #   nm=0 机制对照 → 隔离"噪声"的开销；
        #   非 DP 同预算基线 → "DP 一共"的开销（含 Poisson + 裁剪）。
        # 只报后者会把机制开销算到噪声头上，而 plan §7 明确警告过
        # "把梯度裁剪当成差分隐私"。
        for key, items in _sorted_groups(groups, codes):
            if key[7] is None or key[7][0] <= 0:
                continue
            zero_key = ("fedavg", key[1], None, 0.0, key[4], key[5], None,
                        (0.0, key[7][1], float("inf")))
            if zero_key in groups:
                res = _paired(items, groups[zero_key])
                if res is not None:
                    zm = statistics.mean(r["final_mae"] for r in groups[zero_key])
                    print(_fmt_paired(tag(key), f"{codes[zero_key]} nm=0 机制对照",
                                      res, zm))
            avg_key = ("fedavg", key[1], None, 0.0, key[4], key[5], None, None)
            if avg_key not in groups:
                print(f"  {tag(key)} − FedAvg: 缺同预算的非 DP 基线，跳过")
                continue
            res = _paired(items, groups[avg_key])
            if res is None:
                print(f"  {tag(key)} − FedAvg: 共同 seed 不足 2 个，跳过")
                continue
            avg_mae = statistics.mean(r["final_mae"] for r in groups[avg_key])
            print(_fmt_paired(tag(key), f"{codes[avg_key]} 同预算非 DP", res, avg_mae))

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
