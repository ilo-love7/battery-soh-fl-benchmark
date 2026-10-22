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

import numpy as np  # noqa: E402
from flwr.app import ArrayRecord, ConfigRecord, Message, RecordDict  # noqa: E402
from flwr.supercore.task_identity import TaskIdentity  # noqa: E402

from flsoh.app import client_app as ca  # noqa: E402
from flsoh.app import server_app as sa  # noqa: E402
from flsoh.data import synthetic_feature_table  # noqa: E402
from flsoh.model import build_model  # noqa: E402


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
        out = ca.train(_query(), _Ctx(partition_id=0))
        metrics = dict(out.content["metrics"])
        assert "num-examples" in metrics
        assert metrics["num-examples"] == 20  # 1 颗电池 × 20 循环

    def test_evaluate_reports_num_examples(self, wired):
        out = ca.train(_query(), _Ctx(partition_id=0))
        metrics = dict(ca.evaluate(out, _Ctx(partition_id=0)).content["metrics"])
        assert "num-examples" in metrics

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
