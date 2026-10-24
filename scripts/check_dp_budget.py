#!/usr/bin/env python3
"""DP 预算可行性检查（docs/plan.md §11.3 的「DP 可用性」判据）。

    python3 scripts/check_dp_budget.py --config configs/mvp.toml

## 为什么需要这个脚本

plan §11.3 的判据是「ε ≤ 8 时劣化 ≤30% 支持 H4」。这句话读起来像
「nm 取小一点就能既私密又好用」，但 **ε 不是自由变量**——它由
``(noise_multiplier, 抽样率 q, 步数)`` 三个量决定，而后两个由训练协议
钉死：

    q     = 1 / 每 epoch 批次数 = 1 / ceil(每客户端行数 / batch_size)
    步数  = num_rounds × local_epochs × 每 epoch 批次数

本仓库每客户端只有 5 颗电池 × 16 训练循环 = 80 行、``batch_size=64``，
于是 q = 1/2 = 0.5，100 轮 × le=3 → 600 步。**q=0.5 意味着 subsampling
几乎没有放大作用**（放大小 q 才有用），ε 很快冲到几十几百：

    noise_multiplier=1.0、600 步、q=0.5  →  ε ≈ 152
    noise_multiplier=8.0、600 步、q=0.5  →  ε ≈ 7.8

所以「ε ≤ 8」在这个协议下**等价于「噪声倍数 ≈ 8」**，而 nm=8 时加在
批均值梯度上的噪声 Std = 8 × max_grad_norm / 40 = 0.2，与信号（实测
0.147–0.223）同量级。H4 的两半在这里迎面撞上：**要 ε ≤ 8 就要把噪声
加到和信号一样大**。这不是实现问题，是数据规模问题。

不先算这一步就扫 nm，会出现两种浪费：要么扫的三档全都 ε>8（判据
用不上），要么为了压 ε 把 nm 调到 8 而没意识到那已经是在测「模型
还能不能动」。这个脚本把话说在前面。

## 与 flsoh.protocol 的分工

``flsoh/protocol.py`` 约定纯标准库（§3.2），而算 ε 要 opacus 的 RDP
会计器（scipy）。所以那边只做结构校验，这边做数值校验。

退出码：0 = 至少有一个正的扫描点达到 ε ≤ acceptance.dp_epsilon_max；
1 = 达不到（或配置读不出来）。**达不到不是错误，是结论**——plan §12
的 R5 早就写了「DP 下模型完全学不动……这就是结论之一，不硬凑」，
所以退出码 1 只表示「这份协议的 DP 判据不可用」，调用方要如实报。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flsoh.protocol import ProtocolError, load_config  # noqa: E402


def _batches_per_epoch(rows: int, batch_size: int) -> int:
    """每 epoch 的批次数。``drop_last=False``（DataLoader 默认）。

    80 行 / batch 64 → 2 批（64 + 16）。**最后那个 16 行的小批必须算进来**：
    漏掉它会把批次数当成 1，抽样率翻倍、ε 直接算错——而训练照跑、
    曲线照出，只是预算报错了。
    """
    if rows <= 0:
        raise ValueError(f"每客户端行数必须为正：{rows}")
    return math.ceil(rows / batch_size)


def epsilon_for(noise_multiplier: float, q: float, steps: int, delta: float) -> float:
    """用 opacus 的 RDP 会计器算 ε。

    单独一个函数是为了能单测：这条链路（q、步数 → ε）是整个 M4 判据的
    地基，算错的话后面所有结论都建立在错的横坐标上。

    逐步 ``step()`` 而不是拼一个 ``num_steps=steps`` 的记录：RDP 对
    同 (σ, q) 的多步是**可加**的，两种写法结果相同，但逐步写法与
    ``DPOptimizer`` 每次 ``step()`` 记一笔的路径逐字对应——审计时能
    对上账，不用担心"批量记账和逐笔记账不是一回事"。
    """
    from opacus.accountants import RDPAccountant

    acc = RDPAccountant()
    for _ in range(steps):
        acc.step(noise_multiplier=noise_multiplier, sample_rate=q)
    return float(acc.get_epsilon(delta=delta))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DP 隐私预算可行性检查")
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--rows-per-client",
        type=int,
        default=None,
        help="每客户端训练行数；默认从 partitions.cells_per_client 与 "
             "每颗电池的训练循环数推。推不出来时必须显式给——"
             "猜一个数会让整张预算表失效",
    )
    ap.add_argument(
        "--cycles-per-cell",
        type=int,
        default=16,
        help="每颗电池进训练集的循环数（实测 20 循环按 80/20 切 → 16）",
    )
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ProtocolError as exc:
        print(f"FAIL {args.config}: {exc}")
        return 2

    if not cfg.get("privacy", {}).get("enabled"):
        print(f"OK {args.config}: privacy.enabled=false，本仓库只做 L1（联邦本身）")
        return 0

    noise = cfg["privacy"].get("noise_multiplier") or []
    delta = float(cfg["privacy"].get("target_delta", 1e-5))
    norm = float(cfg["privacy"].get("max_grad_norm", 1.0))
    eps_max = float(cfg.get("acceptance", {}).get("dp_epsilon_max", 8.0))
    rounds = int(cfg["train"]["num_rounds"])
    le = int(cfg["train"]["local_epochs"])
    batch = int(cfg["train"].get("batch_size", 64))

    k_list = cfg.get("partitions", {}).get("cells_per_client") or []
    n_train_cells = int(cfg["data"]["n_cells"]) - int(cfg["partitions"]["test_cells"])
    rows_per_client = args.rows_per_client
    k_used = None
    if rows_per_client is None:
        if not k_list:
            print("FAIL 推不出每客户端行数：partitions.cells_per_client 为空，"
                  "请显式传 --rows-per-client")
            return 2
        # 取最小的一档 k（每客户端电池最少、行数最少）——那是 q 最大、
        # 预算最紧的一档，报它才不会低估代价。
        k_used = min(k_list)
        # k 是**每客户端电池数**，不是客户端数。这里搞反过一版：写成
        # n_train_cells // k 得到 23 颗/客户端、368 行，于是 q 从 0.5 掉到
        # 0.167、ε 从 152 掉到 77——预算表整个偏乐观，而它看起来完全正常。
        rows_per_client = k_used * args.cycles_per_cell
    nbe = _batches_per_epoch(rows_per_client, batch)
    q = 1.0 / nbe
    steps = rounds * le * nbe

    print(f"{args.config}")
    k_txt = f"k={k_used}" if k_used is not None else "--rows-per-client 指定"
    print(f"  每客户端 {rows_per_client} 行（{n_train_cells} 训练电池，{k_txt}，"
          f"每颗 {args.cycles_per_cell} 训练循环）")
    print(f"  batch_size={batch} → 每 epoch {nbe} 批 → 抽样率 q={q:g}")
    print(f"  步数 = {rounds} 轮 × {le} 本地 epoch × {nbe} 批 = {steps}")
    print(f"  δ={delta:g}  max_grad_norm={norm:g}  判据 ε ≤ {eps_max:g}")
    print()
    print(f"  {'noise_multiplier':>16} {'ε':>12} {'达标':>6}  噪声Std(批均值)")

    ok = []
    for nm in noise:
        if nm <= 0:
            print(f"  {nm:>16g} {'inf':>12} {'—':>6}  （对照组，不加噪）")
            continue
        eps = epsilon_for(nm, q, steps, delta)
        # 噪声 Std（加在批均值梯度上）= nm × max_grad_norm / 期望批大小。
        # 期望批大小 = 行数 × q。这个数才是与信号直接可比的量。
        exp_bs = rows_per_client * q
        noise_std = nm * norm / exp_bs
        good = eps <= eps_max
        if good:
            ok.append(nm)
        print(f"  {nm:>16g} {eps:>12.2f} {'是' if good else '否':>6}  {noise_std:.4f}")

    print()
    if not ok:
        print(f"FAIL 没有任何正的扫描点达到 ε ≤ {eps_max:g}。")
        print("  这不是脚本坏了：ε 由 (nm, q, 步数) 决定，q 与步数由训练协议")
        print("  钉死。要让 ε 降下来只有三条路——加大 nm（效用更差）、")
        print("  减步数（减轮数或减 local_epochs，训练量更少）、")
        print("  减 q（减小 batch_size，但每 epoch 批次数变多、总步数反而上升）。")
        print("  plan §12 R5：DP 下学不动就是结论之一，如实报「该规模下不可用」。")
        return 1
    print(f"OK 达到 ε ≤ {eps_max:g} 的扫描点：{ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
