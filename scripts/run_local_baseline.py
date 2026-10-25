#!/usr/bin/env python3
"""中心化（C0）与本地独立（C1）基线——docs/plan.md §11.2。

这两个基线**不经过联邦运行时**，所以单独一个脚本：

- **C0 中心化上限**：把所有客户端的训练电池并起来训同一个模型。
- **C1 本地独立训练**：每个客户端只用自己的数据训，不聚合。

把它们从 `flwr run` 里拆出来的理由：C1 就是 N 个独立训练，为它拉起整套
SuperLink 是纯浪费；而 C0 是上限参考线，必须先拿到它——否则联邦掉点时分不清
是"本来就该掉"还是"代码有 bug"（plan.md §9.4 的顺序不能乱）。

结果按 plan.md §9.3 的 schema 落盘：没跑出来留 ``null``，不许填 0，
也不许删字段。

    python3 scripts/run_local_baseline.py --config configs/mvp.toml \
        --features data/mit-stanford/features.csv --out results/
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from flsoh.data import (  # noqa: E402
    CellDataset,
    FeatureTable,
    LocalStandardizer,
    label_offset,
    load_feature_table,
    make_global_test_loader,
)
from flsoh.featurespec import read_feature_version  # noqa: E402
from flsoh.model import MLP, build_model  # noqa: E402
from flsoh.protocol import (  # noqa: E402
    ProtocolError,
    has_errors,
    load_config,
    validate_config,
)


def _evaluate(model: MLP, loader) -> dict:
    """统一测试集上的 MAE / RMSE / 样本数，在 SOH 原始尺度上。

    标签在 :class:`CellDataset` 里中心化过。MAE/RMSE 平移不变，所以中心化
    不改变这两个数；显式加回偏移是为了让 pred 处于 SOH 尺度（将来报偏差、
    R² 时口径才正确）。偏移从 loader 自己读，不靠调用方传。
    """
    offset = label_offset(loader)
    model.eval()
    se = ae = n = 0.0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).squeeze(-1) + offset
            target = y + offset
            se += float(((pred - target) ** 2).sum())
            ae += float((pred - target).abs().sum())
            n += x.shape[0]
    return {
        "mae": ae / max(n, 1),
        "rmse": (se / max(n, 1)) ** 0.5,
        "n_test": int(n),
    }


def _make_optimizer(model, name: str, lr: float):
    """与 client_app 用同一套选择逻辑，避免基线和联邦走不同优化器。

    默认 adam：plan 的 ``SGD lr=1e-3`` 是在没有真实数据时定的，实测它把
    C0 训成废的（100 ep 测试 MAE 0.0239 vs Adam 0.00415）。
    """
    name = (name or "adam").lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    raise ValueError(f"未知优化器 {name!r}（adam/sgd）")


def _train(model, loader, *, epochs: int, lr: float, optimizer: str = "adam") -> float:
    opt = _make_optimizer(model, optimizer, lr)
    loss_fn = torch.nn.MSELoss()
    model.train()
    last = float("nan")
    for _ in range(epochs):
        total, seen = 0.0, 0
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x).squeeze(-1), y)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * x.shape[0]
            seen += x.shape[0]
        last = total / max(seen, 1)
    return last


def _result(run_id: str, cfg_path: str, strategy: str, *, cells, seed, epochs, lr,
            mae, rmse, wall, notes="", features: str | None = None) -> dict:
    """plan.md §9.3 的固定字段。缺的留 null，不删键。"""
    return {
        "run_id": run_id,
        "config": cfg_path,
        "strategy": strategy,
        "n_clients": len(cells) if isinstance(cells, (list, tuple)) else None,
        "scheme": None,
        "alpha": None,
        "seed": seed,
        "flwr_version": None,   # 基线不走 flwr 运行时，如实留空
        "torch_version": torch.__version__,
        # 线程数要落盘：它改浮点累加顺序，实测 C0 在 24 线程与 1 线程下
        # 差 2.1%（0.0015666 vs 0.0015997）。不记这一项，换台核数不同的
        # 机器复现就永远对不上，而且没人知道原因。
        "torch_num_threads": torch.get_num_threads(),
        "python_version": platform.python_version(),
        # 见 run_federated.py 同名字段：让产物自报特征口径。
        "feature_version": (
            read_feature_version(features) if features else None
        ),
        "final_mae": mae,
        "final_rmse": rmse,
        "rounds_to_target": None,
        "curve": [],
        "wall_clock_s": wall,
        "epochs": epochs,
        "learning_rate": lr,
        "notes": notes,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="跑 C0/C1 基线（不经联邦运行时）")
    ap.add_argument("--config", required=True)
    ap.add_argument("--features", required=True, help="特征表 CSV/parquet")
    ap.add_argument("--partition-json", required=True,
                    help="check_partition.py --emit 的产物，含 test_cells")
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=None,
                    help="默认取 train.baseline_epochs（按 epoch 折算，plan.md §11.2）")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--optimizer", default=None, choices=["adam", "sgd"],
                    help="默认取 train.optimizer；adam 是实测可用的那个")
    ap.add_argument("--seeds", default="",
                    help="逗号分隔的 seed 列表，覆盖配置里的 seeds.values")
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ProtocolError as exc:
        print(f"FAIL {args.config}: {exc}")
        return 2
    if has_errors(validate_config(cfg)):
        print(f"FAIL {args.config}: 协议有 ERROR，先跑 check_protocol.py")
        return 2

    try:
        table = FeatureTable.from_frame(load_feature_table(args.features))
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL 读特征表：{exc}")
        return 2

    payload = json.loads(Path(args.partition_json).read_text(encoding="utf-8"))
    partition = payload["partition"]
    test_cells = payload["test_cells"]

    # 训练电池 = 所有客户端电池；与 test_cells 不相交由 check_partition 保证，
    # 这里再断言一次（plan.md §6.2 规则 2）
    train_cells = sorted({c for cells in partition.values() for c in cells})
    if set(train_cells) & set(test_cells):
        print("FAIL 训练电池与全局测试电池有交集，先跑 check_partition.py --partition-json")
        return 1

    # 基线是单模型训练，与联邦的"100 轮 × local_epochs"不是一回事。
    # 实测 300 ep 是甜点：100 ep 欠拟合（0.00415），500 ep 过拟合（0.00406 回升），
    # 300 ep 得 0.00155。所以单独一个配置项，不从 num_rounds 推。
    epochs = args.epochs or int(cfg["train"].get("baseline_epochs", 300))
    lr = args.lr or float(cfg["train"]["learning_rate"])
    optimizer = args.optimizer or str(cfg["train"].get("optimizer", "adam"))
    batch = int(cfg["train"].get("batch_size", 64))
    # --seeds 覆盖配置里的列表：补 seed 时不该重跑已有的那几个
    # （跑一次 C0 是 50 s，重跑纯属浪费；联邦那边一次 200 s 更浪费）。
    if args.seeds:
        seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    else:
        seeds = list(cfg["seeds"]["values"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_loader, std_strategy = make_global_test_loader(table, test_cells, strategy="test")

    # ---- C0 中心化上限 ----
    # C0 用全局统计量标准化是**允许的**：它本来就是"数据全在一处"的上限参考，
    # 不参与隐私比较。这与联邦场景禁止全局标准化不矛盾。
    scaler = LocalStandardizer().fit(table, train_cells)
    c0_loader = DataLoader(
        CellDataset(table, train_cells, scaler), batch_size=batch, shuffle=True
    )

    for seed in seeds:
        t0 = time.time()
        model = build_model(table.input_dim, seed=seed)
        _train(model, c0_loader, epochs=epochs, lr=lr, optimizer=optimizer)
        wall = round(time.time() - t0, 3)
        m = _evaluate(model, test_loader)
        run = _result(
            f"c0__centralized__seed{seed}", args.config, "centralized",
            cells=None, seed=seed, epochs=epochs, lr=lr,
            mae=m["mae"], rmse=m["rmse"], wall=wall, features=args.features,
            notes=(
                f"test_standardizer={std_strategy}; 全局统计量（C0 允许）; "
                f"训练电池 {len(train_cells)} 颗，单模型（无客户端概念，n_clients 留空）; "
                f"optimizer={optimizer}; label_center=SOH_CENTER"
            ),
        )
        (out_dir / f"{run['run_id']}.json").write_text(
            json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"C0 seed={seed}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  {wall}s")

    # ---- C1 本地独立训练 ----
    for seed in seeds:
        maes, rmses = [], []
        t0 = time.time()
        for cid, cells in sorted(partition.items()):
            # C1 每个客户端用自己的统计量——与联邦时一致，否则下限参考线失真
            local = LocalStandardizer().fit(table, cells)
            loader = DataLoader(
                CellDataset(table, cells, local), batch_size=batch, shuffle=True
            )
            model = build_model(table.input_dim, seed=seed)
            _train(model, loader, epochs=epochs, lr=lr, optimizer=optimizer)
            m = _evaluate(model, test_loader)
            maes.append(m["mae"])
            rmses.append(m["rmse"])
        wall = round(time.time() - t0, 3)
        # 报各客户端指标的均值。注意：这不是"平均模型"，是"平均指标"，
        # 两者不可混谈（plan.md §11.2）
        run = _result(
            f"c1__local__seed{seed}", args.config, "local",
            cells=list(partition), seed=seed, epochs=epochs, lr=lr,
            mae=sum(maes) / len(maes), rmse=sum(rmses) / len(rmses), wall=wall,
            features=args.features,
            notes=(
                f"test_standardizer={std_strategy}; "
                f"{len(partition)} 个客户端各训各的，指标为均值（非平均模型）; "
                f"optimizer={optimizer}"
            ),
        )
        (out_dir / f"{run['run_id']}.json").write_text(
            json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"C1 seed={seed}  MAE={run['final_mae']:.4f}  {wall}s  ({len(partition)} clients)")

    print(f"\n结果已写入 {out_dir}/。联邦实验的相对劣化以 C0 为分母（plan.md §11.3）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
