"""partition 模块测试：划分的确定性、覆盖率，以及五个检查码各自能触发。"""

import pytest

from flsoh.partition import (
    P_COVER,
    P_COUNT,
    P_EMPTY,
    P_LEAK,
    P_TEST,
    P_UNKNOWN,
    PartitionError,
    check_partition,
    client_ids,
    client_order,
    expected_n_clients,
    make_cell_ids,
    partition_by_size,
    partition_dirichlet,
    partition_iid,
    partition_sizes,
    split_test_cells,
)


def _flat(partition):
    return [cell for cells in partition.values() for cell in cells]


class TestMakeCellIds:
    def test_count_and_padding(self):
        cells = make_cell_ids(124)
        assert len(cells) == 124
        assert cells[0] == "cell-000"
        assert cells[-1] == "cell-123"

    def test_lexicographic_order_matches_numeric(self):
        # 宽度不足时 cell-9 会排在 cell-10 之后，日志里看顺序会错
        cells = make_cell_ids(12)
        assert cells == sorted(cells)

    def test_rejects_non_positive(self):
        with pytest.raises(PartitionError):
            make_cell_ids(0)


class TestClientIds:
    def test_shape(self):
        assert client_ids(3) == ["client-0", "client-1", "client-2"]

    def test_rejects_zero(self):
        with pytest.raises(PartitionError):
            client_ids(0)


class TestSplitTestCells:
    def test_disjoint_and_complete(self):
        cells = make_cell_ids(124)
        train, test = split_test_cells(cells, 24, seed=0)
        assert len(train) == 100
        assert len(test) == 24
        assert set(train) & set(test) == set()
        assert set(train) | set(test) == set(cells)

    def test_deterministic(self):
        cells = make_cell_ids(124)
        assert split_test_cells(cells, 24, 0) == split_test_cells(cells, 24, 0)

    def test_different_seed_gives_different_holdout(self):
        cells = make_cell_ids(124)
        _, a = split_test_cells(cells, 24, 0)
        _, b = split_test_cells(cells, 24, 1)
        assert set(a) != set(b)

    def test_rejects_holdout_covering_everything(self):
        cells = make_cell_ids(10)
        with pytest.raises(PartitionError):
            split_test_cells(cells, 10, 0)

    def test_zero_holdout_allowed(self):
        cells = make_cell_ids(10)
        train, test = split_test_cells(cells, 0, 0)
        assert len(train) == 10 and test == []


class TestExpectedNClients:
    def test_divide_evenly(self):
        assert expected_n_clients(124, 24, 5) == 20

    def test_remainder_spread_rounds_up(self):
        # 124 颗、无 holdout、k=5 → 25 个客户端（24 个满的 + 1 个只有 4 颗）
        assert expected_n_clients(124, 0, 5, "spread") == 25

    def test_remainder_drop_rounds_down(self):
        assert expected_n_clients(124, 0, 5, "drop") == 24

    def test_rejects_unknown_remainder(self):
        with pytest.raises(PartitionError):
            expected_n_clients(124, 0, 5, "keep")


class TestPartitionBySize:
    def test_every_client_gets_k(self):
        cells = make_cell_ids(100)
        p = partition_by_size(cells, 5, seed=0)
        assert len(p) == 20
        assert all(len(v) == 5 for v in p.values())

    def test_covers_every_cell_exactly_once(self):
        cells = make_cell_ids(100)
        p = partition_by_size(cells, 5, seed=0)
        assert sorted(_flat(p)) == sorted(cells)

    def test_spread_keeps_leftover_cells(self):
        cells = make_cell_ids(104)
        p = partition_by_size(cells, 5, seed=0, remainder="spread")
        assert len(p) == 21
        assert sorted(_flat(p)) == sorted(cells)

    def test_drop_loses_leftover_cells(self):
        cells = make_cell_ids(104)
        p = partition_by_size(cells, 5, seed=0, remainder="drop")
        assert len(p) == 20
        assert len(_flat(p)) == 100  # 4 颗被丢掉，P-COVER 不会报，因为没进 all_cells

    def test_deterministic_given_seed(self):
        cells = make_cell_ids(100)
        assert partition_by_size(cells, 5, 7) == partition_by_size(cells, 5, 7)

    def test_different_seed_shuffles(self):
        cells = make_cell_ids(100)
        assert partition_by_size(cells, 5, 0) != partition_by_size(cells, 5, 1)


class TestPartitionIid:
    def test_balanced_within_one(self):
        cells = make_cell_ids(100)
        p = partition_iid(cells, 20, seed=0)
        sizes = {len(v) for v in p.values()}
        assert sizes == {5}

    def test_uneven_split_differs_by_at_most_one(self):
        cells = make_cell_ids(100)
        p = partition_iid(cells, 3, seed=0)
        sizes = sorted(len(v) for v in p.values())
        assert sizes[-1] - sizes[0] <= 1

    def test_covers_every_cell(self):
        cells = make_cell_ids(100)
        p = partition_iid(cells, 7, seed=3)
        assert sorted(_flat(p)) == sorted(cells)


class TestPartitionDirichlet:
    def test_no_empty_client_even_at_small_alpha(self):
        cells = make_cell_ids(100)
        p = partition_dirichlet(cells, 20, alpha=0.1, seed=0)
        assert all(p.values())

    def test_covers_every_cell(self):
        cells = make_cell_ids(100)
        p = partition_dirichlet(cells, 20, alpha=0.5, seed=0)
        assert sorted(_flat(p)) == sorted(cells)

    def test_small_alpha_is_more_skewed_than_large(self):
        cells = make_cell_ids(200)
        skewed = partition_sizes(partition_dirichlet(cells, 20, 0.1, 0))
        balanced = partition_sizes(partition_dirichlet(cells, 20, 100.0, 0))
        assert max(skewed) > max(balanced)

    def test_deterministic_given_seed(self):
        cells = make_cell_ids(100)
        assert partition_dirichlet(cells, 20, 0.5, 5) == partition_dirichlet(cells, 20, 0.5, 5)

    def test_rejects_non_positive_alpha(self):
        with pytest.raises(PartitionError):
            partition_dirichlet(make_cell_ids(10), 2, 0.0, 0)

    def test_rejects_more_clients_than_cells(self):
        with pytest.raises(PartitionError):
            partition_dirichlet(make_cell_ids(3), 5, 0.5, 0)


class TestClientOrder:
    """flower 仿真运行时的 partition-id 是整数 0..N-1，划分的键是 client-N。

    按索引取客户端时必须按**数字后缀**排序。直接 sorted() 在 N≥10 时会把
    client-10 排到 client-2 前面，每个客户端拿到别人的电池，且不报错。
    """

    def test_numeric_order_not_lexicographic(self):
        p = {f"client-{i}": [f"cell-{i}"] for i in range(12)}
        order = client_order(p)
        assert order == [f"client-{i}" for i in range(12)]
        assert order.index("client-10") > order.index("client-2")

    def test_lexicographic_sort_would_be_wrong(self):
        # 这条测试钉住"bug 确实存在"：朴素 sorted 的顺序是错的
        p = {f"client-{i}": [] for i in range(12)}
        assert sorted(p) != client_order(p)

    def test_handles_arbitrary_suffixes(self):
        p = {"node-3": [], "node-1": [], "node-20": [], "node-2": []}
        assert client_order(p) == ["node-1", "node-2", "node-3", "node-20"]

    def test_suffix_less_ids_sort_last_and_stable(self):
        p = {"alpha": [], "client-2": [], "client-1": []}
        assert client_order(p) == ["client-1", "client-2", "alpha"]

    def test_empty_partition(self):
        assert client_order({}) == []

    def test_order_is_deterministic(self):
        p = {f"client-{i}": [] for i in range(15)}
        assert client_order(p) == client_order(p)


class TestCheckPartitionClean:
    def _clean(self):
        cells = make_cell_ids(124)
        client_cells, test_cells = split_test_cells(cells, 24, 0)
        p = partition_by_size(client_cells, 5, 0)
        return p, cells, test_cells

    def test_valid_partition_has_no_issues(self):
        p, cells, test = self._clean()
        assert check_partition(p, all_cells=cells, test_cells=test, expected_clients=20) == []

    def test_dirichlet_partition_also_clean(self):
        cells = make_cell_ids(124)
        client_cells, test = split_test_cells(cells, 24, 0)
        p = partition_dirichlet(client_cells, 20, 0.5, 0)
        assert check_partition(p, all_cells=cells, test_cells=test, expected_clients=20) == []


class TestCheckPartitionFaults:
    def _setup(self):
        cells = make_cell_ids(20)
        client_cells, test = split_test_cells(cells, 4, 0)
        p = partition_iid(client_cells, 4, 0)
        return p, cells, test

    def _codes(self, issues):
        return [code for code, _ in issues]

    def test_detects_cross_client_leak(self):
        p, cells, test = self._setup()
        stolen = p["client-0"][0]
        p["client-1"].append(stolen)
        issues = check_partition(p, all_cells=cells, test_cells=test)
        assert P_LEAK in self._codes(issues)
        assert stolen in " ".join(m for c, m in issues if c == P_LEAK)

    def test_detects_duplicate_inside_one_client(self):
        p, cells, test = self._setup()
        p["client-2"].append(p["client-2"][0])
        issues = check_partition(p, all_cells=cells, test_cells=test)
        assert P_LEAK in self._codes(issues)

    def test_detects_test_cell_inside_client(self):
        p, cells, test = self._setup()
        p["client-3"].append(test[0])
        issues = check_partition(p, all_cells=cells, test_cells=test)
        assert P_TEST in self._codes(issues)

    def test_detects_empty_client(self):
        p, cells, test = self._setup()
        p["client-1"] = []
        issues = check_partition(p, all_cells=cells, test_cells=test)
        assert P_EMPTY in self._codes(issues)

    def test_detects_uncovered_cell(self):
        p, cells, test = self._setup()
        orphan = p["client-0"].pop()
        issues = check_partition(p, all_cells=cells, test_cells=test)
        assert P_COVER in self._codes(issues)
        assert orphan in " ".join(m for c, m in issues if c == P_COVER)

    def test_detects_client_count_mismatch(self):
        p, cells, test = self._setup()
        issues = check_partition(p, all_cells=cells, test_cells=test, expected_clients=5)
        assert P_COUNT in self._codes(issues)

    def test_detects_unknown_cell_id(self):
        p, cells, test = self._setup()
        p["client-0"].append("cell-999")
        issues = check_partition(p, all_cells=cells, test_cells=test)
        assert P_UNKNOWN in self._codes(issues)

    def test_one_fault_can_report_multiple_codes(self):
        # 把测试电池塞进客户端：同时是 P-TEST，且原客户端少一颗 → P-COVER 不会报，
        # 因为电池仍在 covered 里。这里钉住"不误报 P-COVER"。
        p, cells, test = self._setup()
        p["client-0"].append(test[0])
        codes = self._codes(check_partition(p, all_cells=cells, test_cells=test))
        assert codes == [P_TEST]
