"""电池级数据划分与泄漏检测。

设计约束（docs/plan.md §3.2 模块契约）：

- 只处理「电池 ID → 客户端」的映射。不碰特征、不读原始数据、不依赖
  numpy / pandas / torch。因此这个模块可以在任何机器上跑、可以进 CI。
- 所有随机性都来自显式 seed。同一个 seed 必得同一份划分，否则
  「3 个 seed 的均值±标准差」不可复现。
- 划分原子固定为「一颗电池」（§6.2 规则 1）。同一颗电池的循环不得
  跨客户端——时间序列被截断会让泛化虚高，而且这种虚高看不出来。

这个模块只回答一个问题：哪颗电池去哪。数据怎么切 train/val、特征怎么
标准化，都不在这里做。
"""

from __future__ import annotations

import random
from collections import Counter

# 划分原子。目前只支持 cell；cycle / session 会破坏 §6.2 规则 1。
PARTITION_ATOM = "cell"

# 检查码，与 docs/plan.md §6.3 的表格一一对应。
P_LEAK = "P-LEAK"
P_TEST = "P-TEST"
P_EMPTY = "P-EMPTY"
P_COVER = "P-COVER"
P_COUNT = "P-COUNT"
P_UNKNOWN = "P-UNKNOWN"

ISSUE_CODES = (P_LEAK, P_TEST, P_EMPTY, P_COVER, P_COUNT, P_UNKNOWN)

REMAINDER_POLICIES = ("spread", "drop")


class PartitionError(ValueError):
    """参数非法，划分根本构造不出来。注意：这不是泄漏。"""


def make_cell_ids(n_cells: int, prefix: str = "cell") -> list[str]:
    """生成占位电池 ID（真实 ID 用 --cells-file 从数据集读）。

    宽度按 n_cells 自动定，保证字典序与数值序一致，日志里好读。
    """
    if n_cells <= 0:
        raise PartitionError(f"n_cells 必须为正整数，收到 {n_cells!r}")
    width = max(3, len(str(n_cells - 1)))
    return [f"{prefix}-{i:0{width}d}" for i in range(n_cells)]


def client_ids(n_clients: int) -> list[str]:
    if n_clients <= 0:
        raise PartitionError(f"n_clients 必须为正整数，收到 {n_clients!r}")
    return [f"client-{i}" for i in range(n_clients)]


def split_test_cells(
    cells: list[str], n_test: int, seed: int
) -> tuple[list[str], list[str]]:
    """整颗留出全局测试电池。

    返回 ``(client_cells, test_cells)``。留出哪些只由 ``seed`` 决定，
    调用方必须传 ``partitions.split_seed`` 而不是训练 seed——否则换一个
    训练 seed 就换一个测试集，各条曲线之间不可比。
    """
    cells = list(cells)
    if n_test < 0:
        raise PartitionError(f"test_cells 不能为负，收到 {n_test}")
    if n_test >= len(cells):
        raise PartitionError(
            f"test_cells={n_test} 吃掉了全部 {len(cells)} 颗电池，客户端将无数据可用"
        )
    rng = random.Random(seed)
    shuffled = cells[:]
    rng.shuffle(shuffled)
    return shuffled[n_test:], shuffled[:n_test]


def expected_n_clients(n_cells: int, n_test: int, cells_per_client: int, remainder: str = "spread") -> int:
    """按配置算「应该有几个客户端」。用于 P-COUNT 对账。"""
    if cells_per_client <= 0:
        raise PartitionError(f"cells_per_client 必须为正整数，收到 {cells_per_client!r}")
    if remainder not in REMAINDER_POLICIES:
        raise PartitionError(
            f"remainder 只能是 {REMAINDER_POLICIES}，收到 {remainder!r}"
        )
    available = n_cells - n_test
    if available <= 0:
        return 0
    if remainder == "spread":
        return -(-available // cells_per_client)  # 向上取整
    return available // cells_per_client


def _largest_remainder(weights: list[float], total: int) -> list[int]:
    """把 ``total`` 个整数按 ``weights`` 分下去，保证求和恰好等于 total。"""
    if total == 0:
        return [0] * len(weights)
    raw = [w * total for w in weights]
    counts = [int(x) for x in raw]
    leftover = total - sum(counts)
    # 余数按小数部分从大到小补，避免总是补给 index 0 造成系统性偏差
    order = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[:leftover]:
        counts[i] += 1
    return counts


def _dirichlet_weights(n: int, alpha: float, rng: random.Random) -> list[float]:
    draws = [rng.gammavariate(alpha, 1.0) for _ in range(n)]
    total = sum(draws)
    if total <= 0:  # 数值退化时退回均匀，不抛异常
        return [1.0 / n] * n
    return [d / total for d in draws]


def partition_by_size(
    cells: list[str],
    cells_per_client: int,
    seed: int,
    remainder: str = "spread",
) -> dict[str, list[str]]:
    """方案 A：每客户端固定 k 颗电池。MVP 主方案。"""
    if remainder not in REMAINDER_POLICIES:
        raise PartitionError(
            f"remainder 只能是 {REMAINDER_POLICIES}，收到 {remainder!r}"
        )
    rng = random.Random(seed)
    shuffled = list(cells)
    rng.shuffle(shuffled)

    n_clients = (
        -(-len(shuffled) // cells_per_client)
        if remainder == "spread"
        else len(shuffled) // cells_per_client
    )
    if n_clients == 0:
        raise PartitionError(
            f"{len(shuffled)} 颗电池不够分给一个客户端（k={cells_per_client}）"
        )

    out: dict[str, list[str]] = {}
    for i, cid in enumerate(client_ids(n_clients)):
        out[cid] = shuffled[i * cells_per_client : (i + 1) * cells_per_client]
    return out


def partition_iid(cells: list[str], n_clients: int, seed: int) -> dict[str, list[str]]:
    """方案 IID：打乱后轮流发牌，各客户端样本数最多差 1。

    IID 指的是「电池到客户端的分配与电池属性无关」，不是「各客户端内部
    同分布」——同一颗电池的数据仍然高度自相关。
    """
    rng = random.Random(seed)
    shuffled = list(cells)
    rng.shuffle(shuffled)
    ids = client_ids(n_clients)
    out: dict[str, list[str]] = {cid: [] for cid in ids}
    for i, cell in enumerate(shuffled):
        out[ids[i % n_clients]].append(cell)
    return out


def partition_dirichlet(
    cells: list[str],
    n_clients: int,
    alpha: float,
    seed: int,
) -> dict[str, list[str]]:
    """方案 D：按 Dirichlet(α) 分配电池，α 越小越异质。

    先给每个客户端保底 1 颗再分剩下的。纯 Dirichlet 在 α<1 时经常产出空
    客户端，而空客户端会触发 P-EMPTY 并且让聚合权重变成 0/0——与其让
    检测器事后报警，不如在构造时就不产生。
    """
    if alpha <= 0:
        raise PartitionError(f"alpha 必须为正数，收到 {alpha!r}")
    cells = list(cells)
    if len(cells) < n_clients:
        raise PartitionError(
            f"电池数 {len(cells)} 少于客户端数 {n_clients}，无法保证每个客户端非空"
        )

    rng = random.Random(seed)
    shuffled = cells[:]
    rng.shuffle(shuffled)

    ids = client_ids(n_clients)
    guaranteed = shuffled[:n_clients]
    rest = shuffled[n_clients:]

    weights = _dirichlet_weights(n_clients, alpha, rng)
    extra = _largest_remainder(weights, len(rest))

    out: dict[str, list[str]] = {}
    cursor = 0
    for cid, base, k in zip(ids, guaranteed, extra):
        out[cid] = [base] + rest[cursor : cursor + k]
        cursor += k
    return out


def partition_by_soh_rank(
    cells: list[str],
    n_clients: int,
    cell_soh: dict[str, float],
    seed: int,
) -> dict[str, list[str]]:
    """方案 E：按电池平均 SOH 排序后切成连续分片（pathological non-IID）。

    **这个方案用到了标签，所以它不是可部署的场景，是一个压力测试。**
    它的用途是回答一个前置问题：如果划分真的造出了标签偏斜，FedAvg 会不会
    掉点？

    为什么需要它：2026-10-24 实测发现本仓库原来的 Dirichlet(α=0.5) 划分
    **造出的标签偏斜比 IID 还小**（客户端身份解释的方差 12.2% vs 17.3%），
    因为 N=23 时它退化成「一个客户端拿 21 颗」，而把 21 颗电池合成一个客户端
    会把电池之间的 SOH 差异平均掉。于是「Non-IID 没掉点」这个结论是无效的
    ——处理根本没施加。本方案把偏斜拉到 k=5 时的上界（≈70%）。

    参数
    ----
    cell_soh
        `{cell_id: 该电池的平均 SOH}`。由调用方从特征表算好传进来，
        这个模块本身仍然不依赖 pandas。

    排序后按分片切：`n_clients` 个客户端各拿 `len(cells)//n_clients` 颗，
    余数按 spread 政策分给前几个客户端（与 `partition_by_size` 一致）。
    同一分片内的电池 SOH 相邻，所以每个客户端只见过一个很窄的 SOH 区间。
    """
    cells = list(cells)
    if n_clients <= 0:
        raise PartitionError(f"n_clients 必须为正，收到 {n_clients!r}")
    missing = [c for c in cells if c not in cell_soh]
    if missing:
        raise PartitionError(
            f"{len(missing)} 颗电池没有 SOH（例如 {missing[0]!r}），无法按 SOH 排序"
        )
    if len(cells) < n_clients:
        raise PartitionError(
            f"电池数 {len(cells)} 少于客户端数 {n_clients}，无法保证每个客户端非空"
        )

    # seed 只用来打散「同一 SOH 值」之间的先后，避免 tie 总是落在同一个客户端。
    # 打散在排序**之前**做，排序本身是稳定的。
    rng = random.Random(seed)
    shuffled = cells[:]
    rng.shuffle(shuffled)
    ordered = sorted(shuffled, key=lambda c: cell_soh[c])

    base = len(ordered) // n_clients
    extra = len(ordered) % n_clients
    out: dict[str, list[str]] = {}
    cursor = 0
    for i, cid in enumerate(client_ids(n_clients)):
        take = base + (1 if i < extra else 0)
        out[cid] = ordered[cursor : cursor + take]
        cursor += take
    return out


def check_partition(
    partition: dict[str, list[str]],
    *,
    all_cells: list[str],
    test_cells: list[str] = (),
    expected_clients: int | None = None,
) -> list[tuple[str, str]]:
    """五项泄漏/一致性检查，返回 ``[(code, message)]``，空列表即 PASS。

    这一组检查跑在训练之前。划分错了，后面所有实验都白做，而且从精度上
    看不出来——只会看到一个"效果很好"的假结论。
    """
    issues: list[tuple[str, str]] = []
    all_set = set(all_cells)
    test_set = set(test_cells)

    # P-COUNT：客户端数与配置声明不符 → 报告里的 N 会写错
    if expected_clients is not None and len(partition) != expected_clients:
        issues.append(
            (
                P_COUNT,
                f"配置应产生 {expected_clients} 个客户端，实际 {len(partition)} 个",
            )
        )

    # P-EMPTY：0 颗电池的客户端在该轮没有梯度，聚合权重异常
    for cid in sorted(partition):
        if not partition[cid]:
            issues.append((P_EMPTY, f"{cid} 未分配到任何电池"))

    # P-LEAK：同一颗电池跨客户端 = 训练集泄漏，精度虚高，结论作废
    owner: dict[str, str] = {}
    for cid in sorted(partition):
        for cell in partition[cid]:
            if cell in owner:
                issues.append(
                    (P_LEAK, f"{cell} 同时出现在 {owner[cell]} 与 {cid}")
                )
            else:
                owner[cell] = cid

    # 客户端内部重复：同一颗电池被算两次，等于把它的权重放大
    for cid in sorted(partition):
        dup = sorted(c for c, k in Counter(partition[cid]).items() if k > 1)
        if dup:
            issues.append((P_LEAK, f"{cid} 内部重复分配：{', '.join(dup)}"))

    # P-TEST：全局测试电池出现在客户端里，比 P-LEAK 更难发现
    for cid in sorted(partition):
        leaked = sorted(set(partition[cid]) & test_set)
        if leaked:
            issues.append(
                (P_TEST, f"全局测试电池出现在 {cid}：{', '.join(leaked)}")
            )

    # P-COVER：有电池既不在客户端也不在测试集 → 数据浪费；
    # 若测试集里出现未知电池，则是另一个方向的污染
    covered = set(owner) | test_set
    missing = sorted(all_set - covered)
    if missing:
        issues.append(
            (
                P_COVER,
                f"{len(missing)} 颗电池既不在客户端也不在测试集：{', '.join(missing[:8])}"
                + ("…" if len(missing) > 8 else ""),
            )
        )

    # P-UNKNOWN：出现了不属于该数据集的电池 ID → 多半是读错了数据源
    unknown = sorted((set(owner) | test_set) - all_set)
    if unknown:
        issues.append(
            (
                P_UNKNOWN,
                f"{len(unknown)} 个 ID 不在数据集清单内：{', '.join(unknown[:8])}"
                + ("…" if len(unknown) > 8 else ""),
            )
        )

    return issues


def partition_sizes(partition: dict[str, list[str]]) -> Counter:
    """各客户端电池数分布，供 CLI 打印一行摘要。"""
    return Counter(len(v) for v in partition.values())


def client_order(partition: dict[str, list[str]]) -> list[str]:
    """按**数字后缀**排序的客户端 ID 列表。

    为什么不能直接 ``sorted(partition)``：字符串排序下 ``client-10`` 会排在
    ``client-2`` 前面。N≥10 时这会静默把客户端顺序打乱——flower 仿真运行时
    按 ``partition-id``（整数 0..N-1）索引客户端，顺序一错，每个客户端拿到
    的就是别人的电池，而且不会有任何报错。

    没有数字后缀的 ID 排在最后，按字典序，保证结果稳定可复现。
    """
    import re

    def sort_key(cid: str):
        m = re.search(r"(\d+)$", cid)
        return (0, int(m.group(1)), "") if m else (1, 0, cid)

    return sorted(partition, key=sort_key)
