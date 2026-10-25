#!/usr/bin/env python3
"""端到端联调：模拟会话 → 入库去重 → 特征 → 训练 → 聚合 → 结果 JSON。

    python3 scripts/run_e2e.py \\
        --sessions data/simulated/sessions.jsonl \\
        --truth data/simulated/truth.json \\
        --out results/e2e__sim__seed0.json

## 它验证的是什么

**不是**「方法有效」——模拟数据上的精度没有外部意义。它验证的是**管道接对**：

    原始记录 → 分段 → ΔQ(V) → 标准化 → 联邦训练 → 聚合 → 指标

每一环都有各自的失效方式，而它们的症状**长得一模一样**：指标变差，或者
「指标居然还挺好」。所以这个脚本在每一环都设了闸门，任何一环不过就非零退出，
并把原因打到 stderr——宁可当场失败，也不要产出一张看起来正常的表。

| 环节 | 闸门 | 不过会怎样 |
|---|---|---|
| 入库 | Schema v1 校验（:mod:`flsoh.schema`） | 结构合法、语义相反的数据进训练集 |
| 去重 | 按内容指纹去重 | 重试上传被算成新数据，客户端样本量虚高 |
| 特征 | TV/幅度 ≤ 60（:mod:`check_features`） | 插值把不相干的点配到一起，而形状完全正常 |
| 标签 | 缺真值直接拒绝 | 缺标签的行把损失静默拉向 0 |
| 训练 | 训练电池与测试电池不得相交 | val 指标虚高 |

## 为什么这里的 TV 阈值是 60 而不是实验室路径的 20

TV/幅度 的**噪声本底随「采样密度 × 电压噪声」线性增长**，它不是跨数据源通用的
无量纲量。实测（1000 点网格）：

    ========================== ========== ==========
    数据源                      TV/幅度    最大跳变/幅度
    ========================== ========== ==========
    实验室 MATR（约 0.2 mV）      5.9        0.11
    模拟 App（2 mV、10 s 采样）   12.8       0.19
      └ 同一批的尾部             45.2       0.55
    错配数据（全循环插值）        122.4      0.78
    ========================== ========== ==========

阈值必须**高于噪声本底**，否则闸门测的是噪声而不是错误——那会让每一次合法的
App 数据都被判红，然后所有人学会加 ``--no-check``。代价要说清楚：在 App 路径上
这道闸门对错配数据的余量只有 **2.7 倍**（122 / 45），而实验室路径是 20 倍。
换句话说，**同一道闸门在 App 路径上弱得多**，这是部署差距的一部分，不是可以
四舍五入掉的细节。

## 与实验室路径的关系

App 路径与实验室路径**共用同一份特征代码**（plan §8.2）：本脚本调
:func:`flsoh.ingest.sessions_to_frame`，后者内部调 ``flsoh.segments``，
最终落到 ``flsoh.features.delta_q_curve`` / ``delta_q_features``——就是
``build_features.py`` 调的那两个函数。所以这里跑通意味着「换成真实 App 数据
也走得通」，**不**意味着两条路径的数值相同（采样率、噪声、参考循环都不同，
差距由 ``segments.app_roundtrip_error`` 量化）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from check_features import check_feature_frame, report_is_ok  # noqa: E402
from flsoh.features import FeatureTable, make_v_grid  # noqa: E402
from flsoh.ingest import (  # noqa: E402
    IngestError,
    ingest_sessions,
    read_sessions_jsonl,
    sessions_to_frame,
    truth_from_json,
)


def round_robin(cells: list[str], k: int) -> dict[str, list[str]]:
    """把电池轮流分给 k 个客户端。

    端到端联调刻意用 IID 划分：这里要测的是**管道**，不是异质性效应。用
    sohrank 之类的划分会把两件事混在一起——指标变差时无法区分是管道接错
    还是 Non-IID 造成的。
    """
    if k <= 0:
        raise ValueError(f"k 必须为正，收到 {k}")
    if len(cells) < k:
        raise ValueError(f"只有 {len(cells)} 颗电池，分不出 {k} 个客户端")
    groups: dict[str, list[str]] = {f"client{i}": [] for i in range(k)}
    for i, cell in enumerate(cells):
        groups[f"client{i % k}"].append(cell)
    return groups


def split_test_cells(cells: list[str], fraction: float) -> tuple[list[str], list[str]]:
    """按**固定步长**取测试电池，不用随机数。

    随机留出会让「同一 seed 重跑」与「换 seed 重跑」的差异混进管道本身的变化里，
    而联调要看的是后者。步长取样还有个好处：测试电池在排序上均匀分布，
    不会恰好全是同一批次。
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction 必须在 (0, 1) 内，收到 {fraction}")
    ordered = sorted(cells)
    stride = max(int(round(1.0 / fraction)), 2)
    test = [c for i, c in enumerate(ordered) if i % stride == 0]
    train = [c for c in ordered if c not in set(test)]
    if not test or not train:
        raise ValueError(f"留出 {fraction:.0%} 后训练或测试集为空")
    return train, test


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="端到端联调（模拟会话 → 联邦训练）")
    ap.add_argument("--sessions", type=Path, required=True, help="Schema v1 JSONL")
    ap.add_argument("--truth", type=Path, required=True, help="真值 JSON")
    ap.add_argument("--out", type=Path, default=None, help="结果 JSON 路径")
    ap.add_argument("--clients", type=int, default=4, help="客户端数（默认 4）")
    ap.add_argument("--test-fraction", type=float, default=0.25,
                    help="留作全局测试的电池比例（默认 0.25）")
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--points", type=int, default=1000, help="电压网格点数")
    ap.add_argument("--v-lo", type=float, default=2.0)
    ap.add_argument("--v-hi", type=float, default=3.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tv-ratio", type=float, default=60.0,
                    help="光滑性闸门的 TV/幅度 **中位数**上限（默认 60，"
                         "见下方说明）。这是主判据")
    ap.add_argument("--max-tv-ratio-hard", type=float, default=180.0,
                    help="TV/幅度 的**单行**硬上限（默认 180）。只做兜底："
                         "NaN、整行常数这类个别坏行，中位数看不出来")
    args = ap.parse_args(argv)

    t0 = time.time()

    # ---- 1. 读 + 校验 + 去重 -------------------------------------------------
    try:
        raw_sessions = read_sessions_jsonl(args.sessions)
    except IngestError as exc:
        print(f"FAIL 读会话失败：{exc}", file=sys.stderr)
        return 2
    report = ingest_sessions(raw_sessions)
    print(
        f"[e2e] 入库：{report['n_total']} 份 → 接收 {report['n_accepted']}，"
        f"拒收 {report['n_invalid']}，重复 {report['n_duplicate']}",
        flush=True,
    )
    if report["n_invalid"]:
        print(
            f"FAIL {report['n_invalid']} 份会话违反 Schema v1。"
            "联调可以继续，但**产出的指标不代表任何真实数据**，所以这里直接停。",
            file=sys.stderr,
        )
        for item in report["errors"][:5]:
            if item["errors"]:
                print(f"    {item['session_id']}: {item['errors'][0]}", file=sys.stderr)
        return 1
    if not report["accepted"]:
        print("FAIL 一份合法会话都没有", file=sys.stderr)
        return 1

    # ---- 2. 特征 -------------------------------------------------------------
    grid = make_v_grid(args.v_lo, args.v_hi, args.points)
    try:
        truth_payload = json.loads(args.truth.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL 读真值失败：{exc}", file=sys.stderr)
        return 2
    truth = truth_from_json(truth_payload)
    try:
        frame, dropped = sessions_to_frame(
            report["accepted"], v_grid=grid, truth=truth
        )
    except IngestError as exc:
        print(f"FAIL 特征组装失败：{exc}", file=sys.stderr)
        return 1
    print(
        f"[e2e] 特征：{frame.shape[0]} 行 × {frame.shape[1] - 3} 维，"
        f"{frame['cell_id'].nunique()} 颗电池，跳过 {len(dropped)} 份会话",
        flush=True,
    )
    for message in dropped[:5]:
        print(f"    跳过 {message}", file=sys.stderr)

    # ---- 3. 光滑性闸门 -------------------------------------------------------
    smooth = check_feature_frame(
        frame,
        max_tv_ratio=args.max_tv_ratio,
        max_tv_ratio_hard=args.max_tv_ratio_hard,
    )
    print(
        f"[e2e] 光滑性：TV/幅度 中位 {smooth['tv_ratio_median']:.2f}"
        f"（阈值 {args.max_tv_ratio:.0f}），"
        f"p99 {smooth['tv_ratio_p99']:.2f}，"
        f"最大 {smooth['tv_ratio_max']:.2f}"
        f"（硬上限 {args.max_tv_ratio_hard:.0f}）；"
        f"最大跳变/幅度 中位 {smooth['max_step_ratio_median']:.3f}"
        f"（诊断量，不设闸门）",
        flush=True,
    )
    if not report_is_ok(smooth):
        print(
            "FAIL 特征表不够光滑——这不是精度问题，是插值链上把不相干的点配到了"
            "一起。先查分段（每个循环含充电+放电两段），再查电压网格是否落在"
            "数据值域内。",
            file=sys.stderr,
        )
        for row in smooth["worst"][:5]:
            print(
                f"    {row['cell_id']} 第 {row['cycle']} 个会话: "
                f"TV/幅度 = {row['tv_ratio']:.1f}",
                file=sys.stderr,
            )
        return 1

    # ---- 4. 划分 -------------------------------------------------------------
    table = FeatureTable.from_frame(frame)
    train_cells, test_cells = split_test_cells(table.cells, args.test_fraction)
    partition = round_robin(train_cells, args.clients)
    print(
        f"[e2e] 划分：{args.clients} 个客户端共 {len(train_cells)} 颗训练电池，"
        f"{len(test_cells)} 颗测试电池",
        flush=True,
    )

    # ---- 5. 训练（这一步才需要 torch） ---------------------------------------
    # 先算「常数基线」：把所有测试行都预测成训练集的平均 SOH。
    # **没有这个数，final_mae 就无法判断**——0.067 看起来像个正常的 MAE，
    # 而模拟数据的标签 std 只有 0.032，常数基线的 MAE 是 0.027，也就是
    # 「模型比一条水平线还差 2.5 倍」。这类数字只有和基线放在一起才有意义。
    train_mask = frame["cell_id"].isin(train_cells)
    test_mask = frame["cell_id"].isin(test_cells)
    baseline_mae = float(
        np.abs(frame.loc[test_mask, "soh"] - frame.loc[train_mask, "soh"].mean()).mean()
    )
    print(
        f"[e2e] 常数基线（全部预测成训练集均值）：MAE {baseline_mae:.5f}",
        flush=True,
    )

    try:
        from run_federated import run_federation
    except ImportError as exc:
        print(
            f"FAIL 训练阶段需要 torch/flwr：{exc}\n"
            "      用项目 venv 跑：D:/项目/venvs/flsoh-py312/Scripts/python.exe",
            file=sys.stderr,
        )
        return 3

    try:
        result = run_federation(
            table,
            partition,
            test_cells,
            strategy="fedavg",
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
            log_every=max(args.rounds // 4, 1),
        )
    except Exception as exc:  # noqa: BLE001 - 训练失败要原样报出来
        print(f"FAIL 训练失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        raise

    # ---- 6. 结果 JSON：与既有 run 同格式，外加 e2e 段 -------------------------
    run_id = f"e2e__sim__seed{args.seed}"
    payload = {
        "run_id": run_id,
        "config": "simulated",
        "strategy": result["strategy"],
        "n_clients": result["n_clients"],
        "scheme": "iid",
        "alpha": None,
        "seed": args.seed,
        "flwr_version": None,
        "torch_version": result["torch_version"],
        "final_mae": result["final_mae"],
        "final_rmse": result["final_rmse"],
        "rounds_to_target": None,
        "curve": result["curve"],
        "wall_clock_s": round(time.time() - t0, 3),
        "epochs": result["rounds"] * result["local_epochs"],
        "learning_rate": result["learning_rate"],
        "backend": "e2e-local-loop",
        "test_standardizer": result["test_standardizer"],
        "notes": (
            "模拟会话（非真实数据），端到端联调用；"
            f"网格 {args.points} 点 [{args.v_lo}, {args.v_hi}] V；"
            "标签相对每颗电池的首个会话"
        ),
        # 与 final_mae 放在一起才可判读：模拟数据的标签 std 只有 0.03 量级，
        # 一个「看起来正常」的 MAE 可能比常数基线还差。
        "baseline_mae": baseline_mae,
        "beats_baseline": bool(result["final_mae"] < baseline_mae),
        "e2e": {
            "n_sessions_raw": report["n_total"],
            "n_sessions_accepted": report["n_accepted"],
            "n_sessions_duplicate": report["n_duplicate"],
            "n_sessions_dropped_in_features": len(dropped),
            "n_rows": int(frame.shape[0]),
            "n_features": int(frame.shape[1] - 3),
            "n_cells": int(frame["cell_id"].nunique()),
            "train_cells": train_cells,
            "test_cells": test_cells,
            "tv_ratio_median": smooth["tv_ratio_median"],
            "tv_ratio_max": smooth["tv_ratio_max"],
            "max_step_ratio_median": smooth["max_step_ratio_median"],
            "max_step_ratio_max": smooth["max_step_ratio_max"],
            "max_tv_ratio": args.max_tv_ratio,
            "max_tv_ratio_hard": args.max_tv_ratio_hard,
            "tv_ratio_p99": smooth["tv_ratio_p99"],
            "n_soft_over": smooth["n_soft_over"],
        },
    }

    out = args.out or Path("results") / "e2e" / f"{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")

    print(
        f"[e2e] 完成：MAE {payload['final_mae']:.5f}  RMSE "
        f"{payload['final_rmse']:.5f}  用时 {payload['wall_clock_s']:.0f}s",
        flush=True,
    )
    print(f"[e2e] 结果写入 {out}")
    if not payload["beats_baseline"]:
        print(
            f"FAIL 模型没跑赢常数基线（{payload['final_mae']:.5f} vs "
            f"{baseline_mae:.5f}）。管道是通的——结果 JSON 已写出——但「特征能"
            "恢复 SOH」这句话在这份运行里不成立。常见原因：轮数/本地 epoch 太少、"
            "标签尺度不对、或标准化把信号抹掉了。",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
