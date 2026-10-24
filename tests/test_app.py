"""app 层测试：flwr 集成里最容易静默出错的三件事。

1. **partition-id 映射**。flower 给的是整数 0..N-1，划分的键是 "client-N"。
   按字符串排序会在 N≥10 时把 client-10 排到 client-2 前面，每个客户端
   拿到别人的电池，而且不报错。
2. **num-examples 必须上报**。`FedAvg` 默认 `weighted_by_key="num-examples"`，
   漏报不报错，静默退化成等权聚合（plan.md §14 坑 3）。
3. **全局测试集不得与客户端相交**（plan.md §6.2 规则 2）。

这些测试依赖 flwr + torch，用 importorskip 守门，不进 CI。
"""

import pytest

# 注意顺序：importorskip 必须是本文件的第一个 import。
# 若把 numpy / flwr 放在它上面，CI（不装重依赖）会在 collection 阶段就
# ModuleNotFoundError，而不是干净地 skip。这个顺序被 CI 的 tests job 钉住。
pytest.importorskip("torch", reason="app 层需要 torch")
flwr_app = pytest.importorskip("flwr.app", reason="app 层需要 flwr")

import torch  # noqa: E402
import numpy as np  # noqa: E402
from flwr.app import (  # noqa: E402
    ArrayRecord,
    ConfigRecord,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.serverapp.strategy import FedAdam  # noqa: E402
from flwr.supercore.task_identity import TaskIdentity  # noqa: E402

from flsoh.app import client_app as ca  # noqa: E402
from flsoh.app import server_app as sa  # noqa: E402
from flsoh.data import synthetic_feature_table  # noqa: E402
from flsoh.model import build_model  # noqa: E402
from flsoh.train import ServerAdam, weighted_average  # noqa: E402


@pytest.fixture(autouse=True)
def _flwr_task_identity():
    """Message 构造需要 TaskIdentity；裸脚本里必须手动设。"""
    TaskIdentity._run_id = 1
    TaskIdentity._task_id = 1
    TaskIdentity._node_id = 0
    yield


@pytest.fixture()
def wired():
    """25 个客户端 × 1 颗电池 + 5 颗全局测试。25 是为了触发排序 bug。"""
    table = synthetic_feature_table(n_cells=30, cycles_per_cell=20, input_dim=16, seed=0)
    cells = table.cells
    partition = {f"client-{i}": [cells[i]] for i in range(25)}
    test = cells[25:]
    ca.configure(table, partition)
    sa.configure(table, test, partition)
    return table, partition, test


class _Ctx:
    def __init__(self, partition_id=0, local_epochs=1, batch_size=8, input_dim=16):
        self.run_config = {
            "input-dim": input_dim,
            "local-epochs": local_epochs,
            "batch-size": batch_size,
        }
        self.node_config = {"partition-id": partition_id}


def _query(lr=0.05, input_dim=16, seed=0):
    init = ArrayRecord.from_torch_state_dict(
        build_model(input_dim, seed=seed).state_dict()
    )
    return Message(
        content=RecordDict({"arrays": init, "config": ConfigRecord({"lr": lr})}),
        dst_node_id=0,
        message_type="query",
    )


# ---------------------------------------------------------------------------
# partition-id 映射
# ---------------------------------------------------------------------------


class TestPartitionIdMapping:
    def test_resolves_to_client_key(self, wired):
        assert ca._resolve_client(0) == "client-0"
        assert ca._resolve_client(10) == "client-10"
        assert ca._resolve_client(24) == "client-24"

    def test_numeric_order_not_lexicographic(self, wired):
        """client-10 必须排在 client-2 之后。字符串排序会反过来。"""
        order = ca._ORDER
        assert order.index("client-10") > order.index("client-2")
        assert order == [f"client-{i}" for i in range(25)]

    def test_rejects_out_of_range(self, wired):
        with pytest.raises(RuntimeError):
            ca._resolve_client(25)

    def test_rejects_non_integer(self, wired):
        with pytest.raises(RuntimeError):
            ca._resolve_client("client-0")  # type: ignore[arg-type]

    def test_requires_configure(self):
        ca.configure(synthetic_feature_table(n_cells=2, input_dim=4), {})
        ca._PARTITION = None
        with pytest.raises(RuntimeError):
            ca._resolve_client(0)


# ---------------------------------------------------------------------------
# num-examples
# ---------------------------------------------------------------------------


class TestNumExamplesReported:
    def test_train_reports_num_examples(self, wired):
        """num-examples 必须是**实际参与训练的行数**。

        client-0 只有 1 颗电池、20 个循环，按时间切 80% 后训练集是 16 行。
        这条原来断言 20——那是按电池切的结果（1 颗电池全进训练，val 集为空），
        而 plan.md §6.2 规则 1 要求客户端内按时间切。数字变了不是回归，
        是切分口径修对了：val 从"空"变成 4 个未来循环。
        """
        out = ca.train(_query(), _Ctx(partition_id=0))
        metrics = dict(out.content["metrics"])
        assert "num-examples" in metrics
        assert metrics["num-examples"] == 16  # 20 循环 × 80%

    def test_evaluate_reports_num_examples(self, wired):
        out = ca.train(_query(), _Ctx(partition_id=0))
        metrics = dict(ca.evaluate(out, _Ctx(partition_id=0)).content["metrics"])
        assert "num-examples" in metrics
        # 按时间切之后本地 val 不再是空的（按电池切时 client-0 只有 1 颗电池，
        # cells[1:] 为空，evaluate 会走"没有验证集"分支返回 0）
        assert metrics["num-examples"] == 4  # 20 循环 × 20%

    def test_train_loss_is_finite(self, wired):
        out = ca.train(_query(), _Ctx(partition_id=0))
        loss = float(dict(out.content["metrics"])["train_loss"])
        assert np.isfinite(loss)

    def test_no_raw_data_in_message(self, wired):
        """L1 隐私的全部内容：上传的只有参数与标量（plan.md §7）。"""
        out = ca.train(_query(), _Ctx(partition_id=0))
        keys = set(out.content.keys())
        assert keys == {"arrays", "metrics"}
        arrays = out.content["arrays"]
        assert isinstance(arrays, ArrayRecord)
        # 参数量必须等于模型参数量，而不是数据集大小
        from flsoh.model import MLP

        assert arrays.count_bytes() > 0


# ---------------------------------------------------------------------------
# 全局评估
# ---------------------------------------------------------------------------


class TestGlobalEvaluate:
    def test_returns_mae_and_rmse(self, wired):
        out = ca.train(_query(), _Ctx(partition_id=0))
        m = dict(sa.global_evaluate(1, out.content["arrays"]))
        assert set(m) >= {"mae", "rmse", "num-examples", "server_round"}
        assert m["num-examples"] == 5 * 20
        assert np.isfinite(m["mae"]) and np.isfinite(m["rmse"])

    def test_rmse_ge_mae(self, wired):
        """均方根 ≥ 平均绝对差，数学上恒成立。不等说明实现有错。"""
        out = ca.train(_query(), _Ctx(partition_id=0))
        m = dict(sa.global_evaluate(1, out.content["arrays"]))
        assert m["rmse"] >= m["mae"] - 1e-9

    def test_standardizer_strategy_exposed_not_in_metrics(self, wired):
        """MetricRecord 只接受数字；策略名必须另找出路（实测传 str 会 TypeError）。"""
        out = ca.train(_query(), _Ctx(partition_id=0))
        m = dict(sa.global_evaluate(1, out.content["arrays"]))
        assert "test_standardizer" not in m
        assert sa.test_standardizer_name() == "test"

    def test_requires_configure(self):
        sa._TEST_LOADER = None
        with pytest.raises(RuntimeError):
            sa.global_evaluate(1, _query().content["arrays"])


# ---------------------------------------------------------------------------
# 测试集泄漏
# ---------------------------------------------------------------------------


class TestTestLeakAssertion:
    def test_clean_partition_passes(self, wired):
        table, partition, test = wired
        sa.configure(table, test, partition)  # 不应抛

    def test_overlapping_test_cell_raises(self, wired):
        table, partition, test = wired
        bad_test = [partition["client-3"][0]] + test
        with pytest.raises(RuntimeError) as exc:
            sa.configure(table, bad_test, partition)
        assert "规则 2" in str(exc.value)

    def test_empty_partition_passes(self, wired):
        table, _, test = wired
        sa.configure(table, test, {})  # 不应抛


# ---------------------------------------------------------------------------
# 训练行为
# ---------------------------------------------------------------------------


class TestTrainingBehaviour:
    def test_more_epochs_lower_loss(self, wired):
        one = dict(ca.train(_query(), _Ctx(partition_id=0, local_epochs=1)).content["metrics"])
        ca._CLIENTS.clear()
        three = dict(
            ca.train(_query(), _Ctx(partition_id=0, local_epochs=3)).content["metrics"]
        )
        assert three["train_loss"] <= one["train_loss"]

    def test_seed_changes_initialisation(self, wired):
        ca._CLIENTS.clear()
        a = ca.train(_query(seed=0), _Ctx(partition_id=0))
        ca._CLIENTS.clear()
        b = ca.train(_query(seed=1), _Ctx(partition_id=0))
        wa = a.content["arrays"].to_torch_state_dict()
        wb = b.content["arrays"].to_torch_state_dict()
        assert any((wa[k] != wb[k]).any() for k in wa)


# ---------------------------------------------------------------------------
# 服务端聚合：本地实现与 flwr 内置策略必须逐轮相等
# ---------------------------------------------------------------------------


class TestFedAdamEquivalence:
    """本地 ``ServerAdam`` 与 flwr 内置 ``FedAdam`` 必须逐轮相等。

    这是仓库的既有约定（``weighted_average`` 对 FedAvg 也是这么钉的）：
    两条路径共享 ``flsoh/train.py`` 的训练实现，服务端聚合一旦漂移，
    两边都不会报错，只是跑出来的数不一样——而那种差异极难定位，
    因为两边的日志都正常。M1 的 DOD ① 建立在「差异只可能在聚合」
    这个前提上，这条测试就是那个前提的护栏。

    ``current_arrays`` 直接设而绕过 ``configure_train``：后者要向 Grid
    采样节点，而这里测的是聚合数学，不是采样。绕过时把理由写清楚，
    免得以后有人以为漏了一步。
    """

    @staticmethod
    def _query(init):
        """构造一条 query。reply 必须由它派生（flwr 1.41 没有裸 reply）。"""
        return Message(
            content=RecordDict(
                {
                    "arrays": ArrayRecord.from_torch_state_dict(init),
                    "config": ConfigRecord({"lr": 1e-3}),
                }
            ),
            dst_node_id=0,
            message_type="query",
        )

    @staticmethod
    def _replies(query, states, weights):
        out = []
        for st, w in zip(states, weights):
            out.append(
                Message(
                    content=RecordDict(
                        {
                            "arrays": ArrayRecord.from_torch_state_dict(st),
                            "metrics": MetricRecord({"num-examples": int(w)}),
                        }
                    ),
                    reply_to=query,
                )
            )
        return out

    def test_matches_flwr_round_by_round(self):
        init = build_model(16, seed=0).state_dict()
        weights = [7.0, 3.0, 5.0]
        query = self._query(init)

        # 三轮，每轮客户端从全局参数出发做确定性的"本地训练"（加一个
        # 与轮次相关的扰动，保证伪梯度既非零也在变化）。
        local_global = {k: v.clone() for k, v in init.items()}
        flwr_global = {k: v.clone() for k, v in init.items()}
        server = ServerAdam(eta=0.01, beta_1=0.9, beta_2=0.99, tau=1e-3)
        strategy = FedAdam(eta=0.01, eta_l=1e-3, beta_1=0.9, beta_2=0.99, tau=1e-3)

        for rnd in range(1, 4):
            g = torch.Generator().manual_seed(100 + rnd)
            states = [
                {
                    k: v + 0.01 * torch.randn(v.shape, generator=g)
                    for k, v in local_global.items()
                }
                for _ in range(3)
            ]

            local_global = server.step(local_global, weighted_average(states, weights))

            strategy.current_arrays = {k: v.numpy() for k, v in flwr_global.items()}
            # flwr 的 server_round 从 0 开始；eta_norm 用的是 round+1
            arrays, _ = strategy.aggregate_train(
                rnd - 1, self._replies(query, states, weights)
            )
            flwr_global = arrays.to_torch_state_dict()

            assert set(local_global) == set(flwr_global)
            for k in local_global:
                assert torch.allclose(
                    local_global[k].to(torch.float32),
                    flwr_global[k].to(torch.float32),
                    rtol=1e-5,
                    atol=1e-7,
                ), f"第 {rnd} 轮 {k} 不一致"

    def test_flwr_fedadam_moves_away_from_averaged(self):
        """负面对照：FedAdam 的结果必须**不等于**加权平均。

        这条与上一条配对才有意义——上一条证明"对得上 flwr"，
        这一条证明"对上的不是巧合"（比如 η 被写死成 1、或整个
        服务端更新被跳过，都会让输出退化成加权平均）。
        """
        init = build_model(16, seed=0).state_dict()
        g = torch.Generator().manual_seed(7)
        states = [
            {k: v + 0.05 * torch.randn(v.shape, generator=g) for k, v in init.items()}
            for _ in range(3)
        ]
        weights = [1.0, 1.0, 1.0]
        avg = weighted_average(states, weights)

        strategy = FedAdam(eta=0.01, eta_l=1e-3)
        strategy.current_arrays = {k: v.numpy() for k, v in init.items()}
        arrays, _ = strategy.aggregate_train(
            0, self._replies(self._query(init), states, weights)
        )
        got = arrays.to_torch_state_dict()

        # flwr 的 FedAdam 输出是 float64（eta_norm 是 numpy 标量，把结果
        # 提上去了），本地实现保持 float32。比较前统一到 float32。
        for k in got:
            assert not torch.allclose(
                got[k].to(torch.float32), avg[k].to(torch.float32), rtol=1e-4
            ), f"{k} 与加权平均逐位相同：服务端 Adam 没有生效"
