"""客户端 App（flwr 1.41 范式，docs/plan.md §3.3）。

两条硬约束写在这里而不是文档里：

1. **`num-examples` 必须上报**。`FedAvg.__init__` 的默认
   ``weighted_by_key="num-examples"``。漏报不报错，只会静默退化成等权
   聚合——而等权聚合在量偏斜下是错的（plan.md §14 坑 3）。
2. **原始数据不进 Message**。上传的只有 ``ArrayRecord``（参数）与
   ``MetricRecord``（标量指标）。这是 L1 隐私的全部内容（plan.md §7）。

`partition-id` / `num-partitions` 由仿真运行时注入 `node_config`，
不需要自己传——这是 1.8+ 的范式，与 v1 的 `NumPyClient` 不同。
"""

from __future__ import annotations

import os

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from torch.utils.data import DataLoader

from flsoh.data import (
    ClientData,
    FeatureTable,
    label_offset,
    load_feature_table,
    make_client_loaders_by_time,
)
from flsoh.model import MLP, build_model
from flsoh.partition import client_order
from flsoh.train import evaluate as evaluate_metrics
from flsoh.train import train_epochs

app = ClientApp()

# 由 main 入口在启动时填充。仿真运行时每个进程一个客户端，
# 所以进程级缓存是安全的；但它仍必须在 run_local_baseline.py 里被显式初始化。
_TABLE: FeatureTable | None = None
_PARTITION: dict[str, list[str]] | None = None
_ORDER: list[str] = []
_CLIENTS: dict[str, ClientData] = {}

# 仿真运行时（ray 后端）把 ClientApp 序列化到独立进程，模块级注入不会跟过去。
# 这两个环境变量是子进程唯一能拿到的入口。名字写死在这里，运行脚本照抄。
ENV_FEATURES = "FLSOH_FEATURES"
ENV_PARTITION = "FLSOH_PARTITION"


def _ensure_loaded() -> None:
    """未 ``configure()`` 时，从环境变量把特征表与划分装进来。

    为什么需要这一层：flwr 1.41 的仿真后端把 ``ClientApp`` 序列化到独立
    actor 进程，``configure()`` 设的模块级变量只活在父进程里。子进程
    import 本模块时 ``_TABLE`` 又是 None，于是第一轮训练就抛
    「未 configure()」——而这个错在父进程里完全看不出来。

    加载的仍然是 ``check_partition.py --emit`` 的**产物文件**，不是现场
    重新划分一次。这点很重要：现场划分会让「跑的到底是哪份划分」不可查。
    """
    if _TABLE is not None and _PARTITION is not None:
        return
    feat = os.environ.get(ENV_FEATURES)
    part = os.environ.get(ENV_PARTITION)
    if not feat or not part:
        raise RuntimeError(
            f"client_app 未 configure()，且环境变量 {ENV_FEATURES} / "
            f"{ENV_PARTITION} 未设置。仿真运行请用 scripts/run_federated.py，"
            "它会设好这两个变量。"
        )
    import json
    from pathlib import Path

    table = FeatureTable.from_frame(load_feature_table(feat))
    payload = json.loads(Path(part).read_text(encoding="utf-8"))
    configure(table, payload["partition"])


def configure(table: FeatureTable, partition: dict[str, list[str]], **loader_kwargs) -> None:
    """注入特征表与划分。必须在任何 ``@app.train()`` 之前调用。

    刻意不做懒加载：划分是 experiment 的一部分，必须来自
    `scripts/check_partition.py --emit` 的产物，不能让训练脚本自己现场分一次。
    """
    global _TABLE, _PARTITION, _ORDER, _CLIENTS
    _TABLE = table
    _PARTITION = partition
    # flower 仿真运行时给的 partition-id 是整数 0..N-1，而划分的键是
    # "client-N"。必须按数字后缀排序后才能按索引取——直接 sorted() 会在
    # N≥10 时把 client-10 排到 client-2 前面，每个客户端拿到别人的电池，
    # 且不报错。
    _ORDER = client_order(partition)
    _CLIENTS = {}


def _resolve_client(partition_id) -> str:
    """把 flower 的 partition-id 映射成划分里的客户端键。"""
    if _PARTITION is None:
        raise RuntimeError("client_app 未 configure()：划分缺失")
    try:
        idx = int(partition_id)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"partition-id 不是整数：{partition_id!r}") from exc
    if not 0 <= idx < len(_ORDER):
        raise RuntimeError(
            f"partition-id={idx} 越界：划分只有 {len(_ORDER)} 个客户端"
        )
    return _ORDER[idx]


def _client_data(cid: str, run_config: dict, node_config: dict) -> ClientData:
    if _TABLE is None:
        raise RuntimeError("client_app 未 configure()：特征表缺失")
    if cid not in _CLIENTS:
        cells = _PARTITION[cid]
        # 客户端内按**时间**切 train/val（plan.md §6.2 规则 1）：同一批电池的
        # 前 80% 循环训练、后 20% 验证。这里原来写的是 cells[:split] /
        # cells[split:]，也就是按电池切——注释说的和代码做的是两件事。
        # 按电池切测的是"能不能认出另一颗电池"，不是"能不能外推到未来循环"，
        # 且本地 val 会虚高。
        _CLIENTS[cid] = make_client_loaders_by_time(
            _TABLE,
            cells,
            batch_size=int(run_config.get("batch-size", 64)),
            seed=int(node_config.get("partition-id", 0)),
        )
    return _CLIENTS[cid]


@app.train()
def train(msg: Message, context: Context) -> Message:
    _ensure_loaded()
    model = MLP(int(context.run_config["input-dim"]))
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    cid = _resolve_client(context.node_config["partition-id"])
    data = _client_data(cid, context.run_config, context.node_config)

    local_epochs = int(context.run_config.get("local-epochs", 1))
    lr = float(msg.content["config"]["lr"])
    optimizer = str(context.run_config.get("optimizer", "adam"))
    mu = float(context.run_config.get("proximal-mu", 0.0))

    # FedProx 的近端项需要"本轮开始时的全局参数"。它就是 msg 里带过来的那份，
    # 所以在训练前先克隆下来——训练开始后 model 会被就地更新，再取就晚了。
    global_params = (
        [p.detach().clone() for p in model.parameters()] if mu > 0 else None
    )

    train_loss = train_epochs(
        model,
        data.train_loader,
        epochs=local_epochs,
        lr=lr,
        optimizer=optimizer,
        mu=mu,
        global_params=global_params,
    )

    arrays = ArrayRecord.from_torch_state_dict(model.state_dict())
    metrics = MetricRecord(
        {
            "train_loss": train_loss,
            # 必须上报：FedAvg 默认按这个键加权（plan.md §14 坑 3）
            "num-examples": data.n_train,
        }
    )
    return Message(content=RecordDict({"arrays": arrays, "metrics": metrics}), reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    _ensure_loaded()
    model = MLP(int(context.run_config["input-dim"]))
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    cid = _resolve_client(context.node_config["partition-id"])
    data = _client_data(cid, context.run_config, context.node_config)

    if data.val_loader is None:
        # 没有本地验证集就返回空指标，而不是编一个 0
        return Message(
            content=RecordDict({"metrics": MetricRecord({"num-examples": 0})}),
            reply_to=msg,
        )

    m = evaluate_metrics(model, data.val_loader)
    metrics = MetricRecord({"mae": m["mae"], "num-examples": data.n_val})
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)
