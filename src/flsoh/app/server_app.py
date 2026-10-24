"""服务端 App（flwr 1.41 范式，docs/plan.md §3.3）。

三件事：

1. **按 `proximal_mu` 选策略**。mu=0 走 `FedAvg`，mu>0 走 `FedProx`。
   这不是偷懒——MVP 的 N2 配置就是同一个 App 换一个参数（plan.md §9.4）。
2. **全局测试集独立于所有客户端**（plan.md §6.2 规则 2）。`evaluate_fn`
   用的测试电池必须与 `_PARTITION` 里任何客户端的电池都不相交，
   启动时断言一次。
3. **C1（本地独立训练）不在这里**。它是 N 个单客户端模型各训各的，
   不需要联邦运行时，见 `scripts/run_local_baseline.py`。
"""

from __future__ import annotations

import os

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx
from torch.utils.data import DataLoader

from flsoh.data import FeatureTable, label_offset, make_global_test_loader
from flsoh.model import MLP, build_model

# 与 client_app 共用同一组环境变量名（见 client_app.ENV_FEATURES 的注释）
ENV_FEATURES = "FLSOH_FEATURES"
ENV_PARTITION = "FLSOH_PARTITION"

app = ServerApp()

_TABLE: FeatureTable | None = None
_TEST_CELLS: list[str] | None = None
_TEST_LOADER: DataLoader | None = None
_TEST_STRATEGY: str = "test"
_INPUT_DIM: int | None = None


def configure(
    table: FeatureTable,
    test_cells: list[str],
    partition: dict[str, list[str]],
    *,
    strategy: str = "test",
) -> None:
    """注入特征表、全局测试电池与客户端划分。

    划分泄漏断言放在这里而不是 `main()` 里：`Grid` 对象不暴露 partition
    （实测 `dir(Grid)` 只有 create_message / get_node_ids / get_nodes /
    pull_messages / push_messages / run / send_and_receive / set_run）。
    放 `main()` 里就会变成一个永远查不到东西的空检查——那比没有检查更糟。
    """
    global _TABLE, _TEST_CELLS, _TEST_LOADER, _TEST_STRATEGY, _INPUT_DIM
    _TABLE = table
    _TEST_CELLS = list(test_cells)
    _TEST_LOADER, _TEST_STRATEGY = make_global_test_loader(
        table, test_cells, strategy=strategy
    )
    _INPUT_DIM = table.input_dim

    _assert_no_test_leak(partition)


def _ensure_loaded() -> None:
    """未 ``configure()`` 时，从环境变量加载特征表与划分。

    与 client_app 同一套机制、同一组变量名。server 端在 ``flwr run`` 里
    也是独立进程，模块级注入同样不会跟过去。
    """
    if _TABLE is not None:
        return
    feat = os.environ.get(ENV_FEATURES)
    part = os.environ.get(ENV_PARTITION)
    if not feat or not part:
        raise RuntimeError(
            f"server_app 未 configure()，且环境变量 {ENV_FEATURES} / "
            f"{ENV_PARTITION} 未设置。仿真运行请用 scripts/run_federated.py。"
        )
    import json
    from pathlib import Path

    from flsoh.data import load_feature_table

    table = FeatureTable.from_frame(load_feature_table(feat))
    payload = json.loads(Path(part).read_text(encoding="utf-8"))
    configure(
        table,
        payload["test_cells"],
        payload["partition"],
        strategy=os.environ.get("FLSOH_TEST_STRATEGY", "test"),
    )


def _assert_no_test_leak(partition: dict[str, list[str]]) -> None:
    """全局测试集与客户端数据不相交。这是 §6.2 规则 2 的执行点。

    `check_partition.py` 已经查过一次，但那是查**产物文件**；
    这里查的是**进程里实际用的那个 partition 对象**——两者之间
    可能有人改过代码。便宜，所以再查一次。
    """
    if _TEST_CELLS is None:
        raise RuntimeError("server_app 未 configure()")
    test_set = set(_TEST_CELLS)
    for cid, cells in partition.items():
        overlap = test_set & set(cells)
        if overlap:
            raise RuntimeError(
                f"客户端 {cid} 含全局测试电池 {sorted(overlap)[:5]}："
                "违反 plan.md §6.2 规则 2，指标会虚高"
            )


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """服务器端全局评估：统一测试集上的 MAE/RMSE。

    这是**唯一**该出现在结果表里的指标（plan.md §11.2）。客户端上报的
    本地 val MAE 只用于诊断，两者不可混报。
    """
    if _TEST_LOADER is None:
        raise RuntimeError("server_app 未 configure()")
    if _INPUT_DIM is None:
        raise RuntimeError("server_app 未 configure()：input_dim 缺失")
    model = MLP(_INPUT_DIM)
    model.load_state_dict(arrays.to_torch_state_dict())
    model.eval()

    # 标签在数据层中心化过（CellDataset），这里加回让 pred 处于 SOH 尺度。
    # MAE/RMSE 本身平移不变，所以中心化不改变这两个数；显式还原是为了
    # 让 pred 的量纲正确——将来报偏差、R² 时漏掉会静默偏 1.0。
    offset = label_offset(_TEST_LOADER)
    se, ae, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in _TEST_LOADER:
            pred = model(x).squeeze(-1) + offset
            se += float(((pred - (y + offset)) ** 2).sum())
            ae += float((pred - (y + offset)).abs().sum())
            n += x.shape[0]
    mae = ae / max(n, 1)
    rmse = (se / max(n, 1)) ** 0.5
    # 注意：MetricRecord 只接受 int/float（实测：传 str 直接 TypeError）。
    # 所以 test_standardizer 不放进指标里，改由 test_standardizer_name()
    # 暴露给运行脚本，写进结果 JSON——评估口径必须可查，但不必塞进指标流。
    return MetricRecord(
        {
            "mae": mae,
            "rmse": rmse,
            "num-examples": n,
            "server_round": server_round,
        }
    )


def test_standardizer_name() -> str:
    """全局测试集实际使用的标准化策略。写进结果 JSON，不进 MetricRecord。"""
    return _TEST_STRATEGY


@app.main()
def main(grid: Grid, context: Context) -> None:
    _ensure_loaded()
    if _TABLE is None:
        raise RuntimeError("server_app 未 configure()")

    num_rounds = int(context.run_config["num-server-rounds"])
    lr = float(context.run_config["learning-rate"])
    mu = float(context.run_config.get("proximal-mu", 0.0))
    if _INPUT_DIM is None:
        raise RuntimeError("server_app 未 configure()：input_dim 缺失")

    strategy = FedProx(proximal_mu=mu) if mu > 0 else FedAvg()

    strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord.from_torch_state_dict(
            build_model(_INPUT_DIM, seed=int(context.run_config.get("seed", 0))).state_dict()
        ),
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )
