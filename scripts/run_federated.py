#!/usr/bin/env python3
"""联邦仿真运行器（plan.md §9.2 的实验矩阵）。

## 为什么不用 `flwr run`

flwr 1.41 的仿真运行时只有 **ray** 后端，而 ray 在本机启动即崩：

    Unhandled exception: [json.exception.type_error.316] invalid UTF-8 byte
    String field 'ray.rpc.GcsNodeInfo.node_manager_hostname' contains invalid
    UTF-8 data when serializing a protocol buffer

根因是**主机名是 GBK 编码的中文**（本机 `寒不忧雪`）。raylet 是 C++ 程序，
用 ANSI API 取主机名拿到 GBK 字节（0xBA 开头），塞进 protobuf 时不是合法
UTF-8。`COMPUTERNAME` 环境变量和 `_node_ip_address` 都覆盖不了它，因为
raylet 直接调 Windows API。

所以本文件提供一条**等价实现**的本地循环，聚合语义与 flwr 的 FedAvg /
FedProx 一致（加权按 num-examples、近端项 (μ/2)·‖w−w_glob‖²）。等价性由
`tests/test_fedloop.py` 钉住：同样的输入，本地聚合与手算的加权平均逐位相等，
μ=0 时 FedProx 与 FedAvg 完全相同。

**这不等于 `flwr run` 已验证。** M1 的 DOD ① 仍需要在 Linux/CI 上跑一次
官方运行时才算数——那里主机名是 ASCII。两条路径共享同一份训练与评估实现
（`flsoh/train.py`），所以差异只可能出在聚合，而聚合被单独测了。

用法::

    python scripts/run_federated.py --config configs/mvp.toml \\
        --features data/mit-stanford/features.parquet \\
        --partition-json results/partition-mvp-k5-iid.json \\
        --strategy fedavg --rounds 100 --out results/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from flsoh.data import (  # noqa: E402
    FeatureTable,
    load_feature_table,
    make_client_loaders_by_time,
    make_global_test_loader,
)
from flsoh.model import build_model  # noqa: E402
from flsoh.protocol import (  # noqa: E402
    ProtocolError,
    has_errors,
    load_config,
    validate_config,
)
from flsoh.train import (  # noqa: E402
    ServerAdam,
    evaluate,
    train_epochs,
    weighted_average,
)


def run_federation(
    table: FeatureTable,
    partition: dict[str, list[str]],
    test_cells: list[str],
    *,
    strategy: str = "fedavg",
    mu: float = 0.0,
    rounds: int = 100,
    local_epochs: int = 2,
    lr: float = 1e-3,
    optimizer: str = "adam",
    batch_size: int = 64,
    seed: int = 0,
    test_strategy: str = "test",
    log_every: int = 10,
    server: ServerAdam | None = None,
) -> dict:
    """跑一轮完整的联邦实验，返回结果 dict（plan.md §9.3 的字段）。

    ``strategy`` 决定服务端怎么合并客户端更新：``fedavg`` 是加权平均；
    ``fedprox`` 在客户端侧加近端项 ``(μ/2)·‖w−w_glob‖²``；``fedadam``
    用服务端 Adam（动量跨轮存活）。这样写不是偷懒——如果两套代码各写
    一遍，跑出来不一样时无法判断是策略差异还是实现差异。

    ``fedadam`` 与 ``mu`` 不互斥：近端项在客户端侧、Adam 在服务端侧，
    两者正交。但 μ>0 时必须显式传，避免「想着 FedAvg、实际带着近端项」
    这种静默组合。
    """
    if strategy not in ("fedavg", "fedprox", "fedadam"):
        raise ValueError(f"未知策略 {strategy!r}（fedavg/fedprox/fedadam）")
    if strategy == "fedprox" and mu <= 0:
        raise ValueError("fedprox 需要 mu > 0；mu=0 就是 fedavg，别混着报")
    if strategy == "fedavg" and mu > 0:
        raise ValueError(f"fedavg 不该带 mu={mu}；要近端项请用 --strategy fedprox")

    # 客户端数据：每个客户端只用自己的电池，且**按时间**切 train/val
    clients = {}
    for cid, cells in sorted(partition.items()):
        clients[cid] = make_client_loaders_by_time(
            table, cells, batch_size=batch_size, seed=seed
        )

    test_loader, used_test_strategy = make_global_test_loader(
        table, test_cells, strategy=test_strategy
    )

    # 一个模型对象复用到底：每轮 load_state_dict 换参数，省掉重复构造
    model = build_model(table.input_dim, seed=seed)
    global_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # 服务端优化器只在 fedadam 下存在；其余策略走加权平均。
    # 有效超显在结果 JSON 里记一份——默认值来自 flwr，但"默认"会随
    # 库版本漂移，记下来复现时才有据可查。
    if strategy == "fedadam" and server is None:
        raise ValueError("strategy=fedadam 需要一个 ServerAdam 实例")
    if strategy != "fedadam" and server is not None:
        raise ValueError("只有 fedadam 才接受 ServerAdam；别让它在别的策略下静默失效")

    curve: list[dict] = []
    t0 = time.time()
    for rnd in range(1, rounds + 1):
        states, weights = [], []
        for cid, data in clients.items():
            model.load_state_dict(global_state)
            # FedProx 的近端项要"本轮开始时的全局参数"，必须在训练前克隆
            gp = (
                [p.detach().clone() for p in model.parameters()] if mu > 0 else None
            )
            train_epochs(
                model,
                data.train_loader,
                epochs=local_epochs,
                lr=lr,
                optimizer=optimizer,
                mu=mu,
                global_params=gp,
            )
            states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
            weights.append(data.n_train)

        averaged = weighted_average(states, weights)
        # FedAdam：伪梯度 = 加权平均 − 当前全局，再走服务端 Adam。
        # η=0 时 step 精确返回 averaged，与 fedavg 逐位相同（有测试钉住）。
        global_state = (
            server.step(global_state, averaged) if server is not None else averaged
        )

        model.load_state_dict(global_state)
        m = evaluate(model, test_loader)
        curve.append({"round": rnd, "mae": m["mae"], "rmse": m["rmse"]})
        if log_every and (rnd % log_every == 0 or rnd == 1 or rnd == rounds):
            print(
                f"  round {rnd:>3}/{rounds}  MAE={m['mae']:.5f}  "
                f"RMSE={m['rmse']:.5f}  ({time.time() - t0:.0f}s)",
                flush=True,
            )

    wall = round(time.time() - t0, 3)
    final = curve[-1]
    result = {
        "strategy": strategy,
        "mu": mu,
        "n_clients": len(clients),
        "rounds": rounds,
        "local_epochs": local_epochs,
        "learning_rate": lr,
        "optimizer": optimizer,
        "batch_size": batch_size,
        "seed": seed,
        "final_mae": final["mae"],
        "final_rmse": final["rmse"],
        "curve": curve,
        "wall_clock_s": wall,
        "test_standardizer": used_test_strategy,
        "backend": "local-loop",
        "torch_version": torch.__version__,
    }
    if server is not None:
        # 服务端超显必须落盘：默认值来自 flwr，而"默认"会随库版本漂移。
        # 复现一份 FedAdam 结果时，η/β/τ 少了任何一个都只能靠猜。
        result["server_lr"] = server.eta
        result["server_beta_1"] = server.beta_1
        result["server_beta_2"] = server.beta_2
        result["server_tau"] = server.tau
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="联邦仿真（本地等价实现，见模块 docstring）")
    ap.add_argument("--config", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--partition-json", required=True)
    ap.add_argument("--strategy", default="fedavg",
                    choices=["fedavg", "fedprox", "fedadam"])
    ap.add_argument("--mu", type=float, default=None, help="FedProx 的 μ，默认取配置")
    ap.add_argument("--server-lr", type=float, default=None,
                    help="FedAdam 服务端学习率 η（默认 flwr 的 1e-1）")
    ap.add_argument("--server-beta1", type=float, default=None,
                    help="FedAdam 服务端一阶矩 β1（默认 0.9）")
    ap.add_argument("--server-beta2", type=float, default=None,
                    help="FedAdam 服务端二阶矩 β2（默认 0.99）")
    ap.add_argument("--server-tau", type=float, default=None,
                    help="FedAdam 的自适应程度 τ（默认 1e-3）")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--local-epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--optimizer", default=None, choices=["adam", "sgd"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default=None, help="结果文件名后缀，默认由参数拼出")
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ProtocolError as exc:
        print(f"FAIL {args.config}: {exc}")
        return 2
    if has_errors(validate_config(cfg)):
        print(f"FAIL {args.config}: 协议有 ERROR，先跑 check_protocol.py")
        return 2

    table = FeatureTable.from_frame(load_feature_table(args.features))
    payload = json.loads(Path(args.partition_json).read_text(encoding="utf-8"))
    partition = payload["partition"]
    test_cells = payload["test_cells"]

    # 训练电池与测试电池不得相交（plan.md §6.2 规则 2）。check_partition 查过
    # 产物，这里查的是进程里实际用的这份。
    train_cells = {c for cells in partition.values() for c in cells}
    if train_cells & set(test_cells):
        print("FAIL 训练电池与测试电池有交集，先跑 check_partition.py --partition-json")
        return 1

    rounds = args.rounds or int(cfg["train"]["num_rounds"])
    local_epochs = args.local_epochs or int(cfg["train"].get("local_epochs", 2))
    lr = args.lr or float(cfg["train"]["learning_rate"])
    optimizer = args.optimizer or str(cfg["train"].get("optimizer", "adam"))
    batch_size = int(cfg["train"].get("batch_size", 64))

    if args.mu is not None:
        mu = args.mu
    elif args.strategy == "fedprox":
        # 配置里 proximal_mu 是候选列表，fedprox 必须显式挑一个
        mus = cfg["strategies"].get("proximal_mu", [])
        mu = float(mus[-1]) if mus else 0.1
        print(f"fedprox 未指定 --mu，取配置里的 {mu}")
    else:
        mu = 0.0

    print(
        f"{args.strategy}  N={len(partition)}  rounds={rounds}  "
        f"local_epochs={local_epochs}  lr={lr}  {optimizer}  mu={mu}  seed={args.seed}",
        flush=True,
    )
    server = None
    if args.strategy == "fedadam":
        # 只把显式给了的项传进去，其余用 ServerAdam 的默认（= flwr 的默认）。
        # 用 `is not None` 而不是 `or`：`--server-lr 0` 是合法输入，
        # `0 or default` 会静默把它换成默认值。
        kwargs = {}
        if args.server_lr is not None:
            kwargs["eta"] = args.server_lr
        if args.server_beta1 is not None:
            kwargs["beta_1"] = args.server_beta1
        if args.server_beta2 is not None:
            kwargs["beta_2"] = args.server_beta2
        if args.server_tau is not None:
            kwargs["tau"] = args.server_tau
        server = ServerAdam(**kwargs)
        # 服务端超显打出来：η 选错时曲线会在前几轮就停滞，日志里能看到
        # 实际生效的值，不用回去翻命令行。
        print(
            f"  FedAdam 服务端：eta={server.eta:g}  beta_1={server.beta_1:g}  "
            f"beta_2={server.beta_2:g}  tau={server.tau:g}",
            flush=True,
        )
    result = run_federation(
        table,
        partition,
        test_cells,
        strategy=args.strategy,
        mu=mu,
        rounds=rounds,
        local_epochs=local_epochs,
        lr=lr,
        optimizer=optimizer,
        batch_size=batch_size,
        seed=args.seed,
        test_strategy=str(cfg["train"].get("test_standardizer", "test")),
        server=server,
    )

    scheme = payload.get("scheme") or "unknown"
    alpha = payload.get("alpha")
    # η 进文件名：扫 η 时两个只差服务端学习率的 run 会撞名，而撞名的
    # 表现是后一个静默覆盖前一个——结果目录里少一半数据且不报错。
    tag = args.tag or (
        f"{args.strategy}__{scheme}"
        + (f"{alpha}" if alpha is not None else "")
        + (f"__eta{server.eta:g}" if server is not None else "")
        + f"__seed{args.seed}"
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.json"
    result["run_id"] = tag
    result["config"] = args.config
    result["partition_json"] = str(args.partition_json)
    result["scheme"] = scheme
    result["alpha"] = alpha
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n最终 MAE={result['final_mae']:.5f}  RMSE={result['final_rmse']:.5f}  "
          f"耗时 {result['wall_clock_s']}s")
    print(f"已写入 {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
