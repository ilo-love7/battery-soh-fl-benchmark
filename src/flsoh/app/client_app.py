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

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from torch.utils.data import DataLoader

from flsoh.data import ClientData, FeatureTable, make_client_loaders
from flsoh.model import MLP, build_model
from flsoh.partition import client_order

app = ClientApp()

# 由 main 入口在启动时填充。仿真运行时每个进程一个客户端，
# 所以进程级缓存是安全的；但它仍必须在 run_local_baseline.py 里被显式初始化。
_TABLE: FeatureTable | None = None
_PARTITION: dict[str, list[str]] | None = None
_ORDER: list[str] = []
_CLIENTS: dict[str, ClientData] = {}


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
        # 客户端内按时间切 train/val：取前 80% 循环训练，后 20% 验证。
        # 切分单位是循环而不是电池——同一颗电池的循环不能跨 train/val
        # 之外的边界，但 train/val 之间按时间是允许的（plan.md §6.2 规则 1）。
        split = max(1, int(len(cells) * 0.8))
        _CLIENTS[cid] = make_client_loaders(
            _TABLE,
            cells[:split],
            cells[split:],
            batch_size=int(run_config.get("batch-size", 64)),
            seed=int(node_config.get("partition-id", 0)),
        )
    return _CLIENTS[cid]


def _train_one_round(
    model: MLP, loader: DataLoader, local_epochs: int, lr: float
) -> float:
    """本地训练。返回最后一个 epoch 的平均 loss。

    用 MSE 而不是 MAE：SOH 是回归量，MSE 对离群循环更敏感，
    而 MAE 在接近 0 时梯度不降。两者都要报（plan.md §10.2 表 1）。
    """
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    model.train()
    last = float("nan")
    for _ in range(local_epochs):
        total, seen = 0.0, 0
        for x, y in loader:
            opt.zero_grad()
            pred = model(x).squeeze(-1)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * x.shape[0]
            seen += x.shape[0]
        last = total / max(seen, 1)
    return last


def _mae(model: MLP, loader: DataLoader) -> float:
    model.eval()
    total, seen = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).squeeze(-1)
            total += float((pred - y).abs().sum())
            seen += x.shape[0]
    return total / max(seen, 1)


@app.train()
def train(msg: Message, context: Context) -> Message:
    model = MLP(int(context.run_config["input-dim"]))
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    cid = _resolve_client(context.node_config["partition-id"])
    data = _client_data(cid, context.run_config, context.node_config)

    local_epochs = int(context.run_config.get("local-epochs", 1))
    lr = float(msg.content["config"]["lr"])

    train_loss = _train_one_round(model, data.train_loader, local_epochs, lr)

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

    mae = _mae(model, data.val_loader)
    metrics = MetricRecord({"mae": mae, "num-examples": data.n_val})
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)
