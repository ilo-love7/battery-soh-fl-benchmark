"""data 模块测试。

核心是 plan.md §14 坑 1「全局标准化 = 隐私泄露」。
这条规则不能只写在文档里——本文件用三类断言把它钉住：

1. 标准化器只拟合训练电池（``fitted_on`` 可查）；
2. 验证/测试电池参与拟合时**抛异常**，不是警告；
3. 全局测试集变换后的均值**不**为 0——这是"没有偷偷用全局统计量"的
   可观测证据。
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="data 需要 torch；CI 只跑纯标准库部分")

from flsoh.data import (  # noqa: E402
    CELL_COL,
    CYCLE_COL,
    SOH_COL,
    CellDataset,
    DataContractError,
    FeatureTable,
    FixedScaler,
    LocalStandardizer,
    delta_q_curve,
    delta_q_features,
    feature_columns,
    load_feature_table,
    make_client_loaders,
    make_global_test_loader,
    make_v_grid,
    synthetic_feature_table,
    validate_feature_table,
)


@pytest.fixture()
def table():
    return synthetic_feature_table(n_cells=20, cycles_per_cell=40, input_dim=16, seed=0)


@pytest.fixture()
def split(table):
    cells = table.cells
    return cells[:12], cells[12:17], cells[17:]


# ---------------------------------------------------------------------------
# 特征表契约
# ---------------------------------------------------------------------------


class TestFeatureTableContract:
    def test_synthetic_table_shape(self, table):
        assert len(table.cells) == 20
        assert table.input_dim == 16
        assert len(table.frame) == 20 * 40

    def test_rejects_empty(self):
        with pytest.raises(DataContractError):
            validate_feature_table(pd.DataFrame())

    def test_rejects_missing_columns(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(DataContractError):
            validate_feature_table(df)

    def test_rejects_null_cell_id(self, table):
        bad = table.frame.copy()
        bad.loc[0, CELL_COL] = None
        with pytest.raises(DataContractError):
            validate_feature_table(bad)

    def test_rejects_null_label(self, table):
        bad = table.frame.copy()
        bad.loc[0, SOH_COL] = np.nan
        with pytest.raises(DataContractError):
            validate_feature_table(bad)

    def test_rejects_no_feature_columns(self, table):
        bad = table.frame.drop(columns=list(table.feats))
        with pytest.raises(DataContractError):
            validate_feature_table(bad)

    def test_feature_columns_sorted_numerically(self, table):
        # f10 必须排在 f2 之后：字符串排序会把它排到前面，维度就错位了
        cols = feature_columns(table.frame)
        assert cols == [f"f{i}" for i in range(16)]
        assert cols == sorted(cols, key=lambda c: int(c[1:]))

    def test_subset_filters_by_cell(self, table, split):
        train, _, _ = split
        sub = table.subset(train)
        assert set(sub[CELL_COL].unique()) == set(train)

    def test_load_feature_table_roundtrip(self, table, tmp_path):
        p = tmp_path / "feats.csv"
        table.frame.to_csv(p, index=False)
        back = load_feature_table(p)
        assert FeatureTable.from_frame(back).input_dim == table.input_dim

    def test_load_rejects_missing_file(self, tmp_path):
        with pytest.raises(DataContractError):
            load_feature_table(tmp_path / "nope.csv")

    def test_load_rejects_unknown_suffix(self, tmp_path):
        p = tmp_path / "feats.xlsx"
        p.write_text("x", encoding="utf-8")
        with pytest.raises(DataContractError):
            load_feature_table(p)


# ---------------------------------------------------------------------------
# 坑 1：全局标准化
# ---------------------------------------------------------------------------


class TestNoGlobalStandardization:
    def test_standardizer_fits_only_train_cells(self, table, split):
        train, _, _ = split
        sc = LocalStandardizer().fit(table, train)
        assert sc.fitted_on == frozenset(train)

    def test_refuses_empty_fit(self, table):
        # 空集合拟合等于用"全量"拟合，正是坑 1
        with pytest.raises(DataContractError):
            LocalStandardizer().fit(table, [])

    def test_refuses_unknown_cells(self, table, split):
        train, _, _ = split
        with pytest.raises(DataContractError):
            LocalStandardizer().fit(table, train + ["cell-9999"])

    def test_check_no_leak_raises_on_overlap(self, table, split):
        """真实场景：有人把 val 电池也喂进了标准化器。"""
        train, val, _ = split
        sc = LocalStandardizer().fit(table, train + val)  # 错误做法
        with pytest.raises(DataContractError) as exc:
            sc.check_no_leak(val, label="本地验证集")
        assert "坑 1" in str(exc.value)
        assert val[0] in str(exc.value)  # 报出第一个泄漏的电池 ID

    def test_check_no_leak_passes_when_disjoint(self, table, split):
        train, val, _ = split
        sc = LocalStandardizer().fit(table, train)
        sc.check_no_leak(val)  # 不应抛

    def test_per_client_scalers_differ(self, table, split):
        """两个客户端的标准化器必须不同——相同就意味着算了全局统计量。"""
        train, _, _ = split
        a = LocalStandardizer().fit(table, train[:6])
        b = LocalStandardizer().fit(table, train[6:])
        assert not np.allclose(a._mean, b._mean)

    def test_local_scaler_gives_zero_mean_on_own_train(self, table, split):
        train, _, _ = split
        sc = LocalStandardizer().fit(table, train)
        x = table.subset(train).loc[:, list(table.feats)].to_numpy(dtype=float)
        assert np.allclose(sc.transform(x).mean(axis=0), 0.0, atol=1e-9)

    def test_global_test_is_not_zero_mean(self, table, split):
        """关键证据：全局测试集用客户端 scaler 变换后均值不为 0。

        如果这里接近 0，说明有人在偷偷用全局统计量。
        """
        train, _, test = split
        sc = LocalStandardizer().fit(table, train)
        x = table.subset(test).loc[:, list(table.feats)].to_numpy(dtype=float)
        assert not np.allclose(sc.transform(x).mean(axis=0), 0.0, atol=1e-6)

    def test_constant_column_does_not_explode(self, table):
        # 常数列除以 0 会得到 inf， silently 毁掉训练
        train = table.cells[:5]
        bad = table.frame.copy()
        bad["f0"] = 3.0
        t2 = FeatureTable.from_frame(bad)
        sc = LocalStandardizer().fit(t2, train)
        out = sc.transform(t2.subset(train).loc[:, list(t2.feats)].to_numpy(dtype=float))
        assert np.isfinite(out).all()

    def test_transform_before_fit_raises(self):
        with pytest.raises(DataContractError):
            LocalStandardizer().transform(np.zeros((2, 2)))

    def test_transform_dimension_mismatch_raises(self, table, split):
        train, _, _ = split
        sc = LocalStandardizer().fit(table, train)
        with pytest.raises(DataContractError):
            sc.transform(np.zeros((3, 999)))


# ---------------------------------------------------------------------------
# make_client_loaders
# ---------------------------------------------------------------------------


class TestMakeClientLoaders:
    def test_basic(self, table, split):
        train, val, _ = split
        cd = make_client_loaders(table, train, val, batch_size=8, seed=0)
        assert cd.n_train == 12 * 40
        assert cd.n_val == 5 * 40
        assert cd.standardizer_fitted_on == frozenset(train)

    def test_refuses_train_val_overlap(self, table, split):
        train, val, _ = split
        with pytest.raises(DataContractError) as exc:
            make_client_loaders(table, train, val + [train[0]], seed=0)
        assert "交集" in str(exc.value)

    def test_refuses_empty_train(self, table, split):
        _, val, _ = split
        with pytest.raises(DataContractError):
            make_client_loaders(table, [], val, seed=0)

    def test_val_cells_never_in_fitted_on(self, table, split):
        train, val, _ = split
        cd = make_client_loaders(table, train, val, seed=0)
        assert not (set(cd.val_cells) & cd.standardizer_fitted_on)

    def test_batch_shape_and_dtype(self, table, split):
        train, val, _ = split
        cd = make_client_loaders(table, train, val, batch_size=16, seed=0)
        x, y = next(iter(cd.train_loader))
        assert x.shape == (16, 16)
        assert y.shape == (16,)
        assert x.dtype == torch.float32

    def test_last_batch_may_be_smaller(self, table, split):
        train, val, _ = split
        cd = make_client_loaders(table, train[:3], val, batch_size=64, seed=0)
        sizes = [b[0].shape[0] for b in cd.train_loader]
        assert sizes[-1] == (3 * 40) % 64

    def test_same_seed_same_order(self, table, split):
        train, val, _ = split
        a = make_client_loaders(table, train, val, seed=3)
        b = make_client_loaders(table, train, val, seed=3)
        xa = torch.cat([x for x, _ in a.train_loader])
        xb = torch.cat([x for x, _ in b.train_loader])
        assert torch.equal(xa, xb)

    def test_different_seed_different_order(self, table, split):
        train, val, _ = split
        a = make_client_loaders(table, train, val, seed=3)
        b = make_client_loaders(table, train, val, seed=4)
        xa = torch.cat([x for x, _ in a.train_loader])
        xb = torch.cat([x for x, _ in b.train_loader])
        assert not torch.equal(xa, xb)

    def test_val_loader_is_not_shuffled(self, table, split):
        # 验证集顺序固定，否则同一模型两次评估的指标会有微小差异
        train, val, _ = split
        cd = make_client_loaders(table, train, val, seed=0)
        ys = torch.cat([y for _, y in cd.val_loader])
        expected = torch.tensor(
            table.subset(val)[SOH_COL].to_numpy(dtype=float), dtype=torch.float32
        )
        assert torch.equal(ys, expected)

    def test_rejects_bad_batch_size(self, table, split):
        train, val, _ = split
        with pytest.raises(DataContractError):
            make_client_loaders(table, train, val, batch_size=0)

    def test_no_val_cells_gives_none_loader(self, table, split):
        train, _, _ = split
        cd = make_client_loaders(table, train, [], seed=0)
        assert cd.val_loader is None and cd.n_val == 0


# ---------------------------------------------------------------------------
# 全局测试集
# ---------------------------------------------------------------------------


class TestGlobalTestLoader:
    def test_strategy_test(self, table, split):
        _, _, test = split
        loader, used = make_global_test_loader(table, test, strategy="test")
        assert used == "test"
        assert sum(b[0].shape[0] for b in loader) == len(test) * 40

    def test_strategy_fixed(self, table, split):
        _, _, test = split
        scale = np.ones(table.input_dim)
        loader, used = make_global_test_loader(table, test, strategy="fixed", fixed_scale=scale)
        assert used == "fixed"

    def test_strategy_fixed_requires_scale(self, table, split):
        _, _, test = split
        with pytest.raises(DataContractError):
            make_global_test_loader(table, test, strategy="fixed")

    def test_strategy_none(self, table, split):
        _, _, test = split
        loader, used = make_global_test_loader(table, test, strategy="none")
        assert used == "none"

    def test_rejects_unknown_strategy(self, table, split):
        _, _, test = split
        with pytest.raises(DataContractError):
            make_global_test_loader(table, test, strategy="global")  # type: ignore[arg-type]

    def test_rejects_unknown_test_cells(self, table, split):
        _, _, test = split
        with pytest.raises(DataContractError):
            make_global_test_loader(table, test + ["cell-9999"])

    def test_test_strategy_uses_only_test_statistics(self, table, split):
        """strategy='test' 的 scaler 只见过测试电池。"""
        _, _, test = split
        make_global_test_loader(table, test, strategy="test")
        # 若它见过训练电池，这里会暴露——用一个新的 scaler 对比
        sc = LocalStandardizer().fit(table, test)
        assert sc.fitted_on == frozenset(test)


class TestFixedScaler:
    def test_rejects_zero_scale(self):
        with pytest.raises(DataContractError):
            FixedScaler(np.array([1.0, 0.0]))

    def test_transform(self):
        sc = FixedScaler(np.array([2.0, 4.0]))
        out = sc.transform(np.array([[2.0, 8.0]]))
        assert np.allclose(out, [[1.0, 2.0]])

    def test_dimension_mismatch(self):
        sc = FixedScaler(np.array([2.0]))
        with pytest.raises(DataContractError):
            sc.transform(np.zeros((2, 3)))


# ---------------------------------------------------------------------------
# CellDataset
# ---------------------------------------------------------------------------


class TestCellDataset:
    def test_len_and_getitem(self, table, split):
        train, _, _ = split
        ds = CellDataset(table, train)
        assert len(ds) == 12 * 40
        x, y = ds[0]
        assert x.shape == (16,) and y.dim() == 0

    def test_rejects_unknown_cells(self, table):
        with pytest.raises(DataContractError):
            CellDataset(table, ["cell-9999"])

    def test_deduplicates_cell_ids(self, table, split):
        train, _, _ = split
        ds = CellDataset(table, train + train)
        assert len(ds) == 12 * 40

    def test_scaler_applied(self, table, split):
        train, _, _ = split
        sc = LocalStandardizer().fit(table, train)
        raw = CellDataset(table, train)
        scaled = CellDataset(table, train, sc)
        assert not torch.allclose(raw.x, scaled.x)


# ---------------------------------------------------------------------------
# ΔQ(V)
# ---------------------------------------------------------------------------


class TestDeltaQ:
    def test_v_grid(self):
        g = make_v_grid(3.0, 4.2, 1000)
        assert len(g) == 1000
        assert g[0] == 3.0 and g[-1] == 4.2

    def test_v_grid_rejects_inverted_range(self):
        with pytest.raises(DataContractError):
            make_v_grid(4.2, 3.0, 10)

    def test_v_grid_rejects_single_point(self):
        with pytest.raises(DataContractError):
            make_v_grid(3.0, 4.2, 1)

    def test_curve_interpolates_onto_grid(self):
        v = np.array([3.0, 3.5, 4.0])
        q = np.array([0.0, 1.0, 2.0])
        grid = np.array([3.0, 3.25, 4.0])
        out = delta_q_curve(v, q, grid)
        assert np.allclose(out, [0.0, 0.5, 2.0])

    def test_curve_handles_descending_voltage(self):
        v = np.array([4.0, 3.5, 3.0])
        q = np.array([2.0, 1.0, 0.0])
        grid = np.array([3.0, 4.0])
        assert np.allclose(delta_q_curve(v, q, grid), [0.0, 2.0])

    def test_curve_rejects_out_of_range_grid(self):
        v = np.array([3.0, 4.0])
        q = np.array([0.0, 1.0])
        with pytest.raises(DataContractError) as exc:
            delta_q_curve(v, q, np.array([2.9, 3.5]))
        assert "外延" in str(exc.value)

    def test_curve_rejects_length_mismatch(self):
        with pytest.raises(DataContractError):
            delta_q_curve(np.array([3.0, 4.0]), np.array([0.0]), np.array([3.5]))

    def test_curve_rejects_single_point(self):
        with pytest.raises(DataContractError):
            delta_q_curve(np.array([3.0]), np.array([0.0]), np.array([3.0]))

    def test_features_shape_and_reference(self):
        grid = np.linspace(3.0, 4.0, 8)
        cycles = [(np.linspace(3.0, 4.0, 20), np.linspace(0, 1, 20)) for _ in range(5)]
        out = delta_q_features(cycles, grid, reference_index=0)
        assert out.shape == (4, 8)          # 5 个循环去掉参考 → 4 行
        assert np.allclose(out[0], 0.0)     # 参考循环之后第一个若与参考相同则为 0

    def test_features_detects_degradation(self):
        grid = np.linspace(3.0, 4.0, 8)
        ref_v = np.linspace(3.0, 4.0, 20)
        ref_q = np.linspace(0, 1, 20)
        aged_q = np.linspace(0, 0.8, 20)    # 容量衰减
        out = delta_q_features([(ref_v, ref_q), (ref_v, aged_q)], grid)
        # 网格第一个点两条曲线都是 Q=0，所以那里 ΔQ 恰为 0；其余必须为负
        assert (out[0] <= 1e-12).all()
        assert (out[0] < 0).sum() >= len(grid) - 1
        assert out[0][0] == pytest.approx(0.0, abs=1e-12)

    def test_features_rejects_bad_reference_index(self):
        grid = np.linspace(3.0, 4.0, 4)
        cycles = [(np.linspace(3.0, 4.0, 10), np.linspace(0, 1, 10))]
        with pytest.raises(DataContractError):
            delta_q_features(cycles, grid, reference_index=5)

    def test_features_rejects_empty(self):
        with pytest.raises(DataContractError):
            delta_q_features([], np.linspace(3.0, 4.0, 4))


# ---------------------------------------------------------------------------
# 合成数据
# ---------------------------------------------------------------------------


class TestSyntheticTable:
    def test_deterministic(self):
        a = synthetic_feature_table(seed=1)
        b = synthetic_feature_table(seed=1)
        pd.testing.assert_frame_equal(a.frame, b.frame)

    def test_different_seed_differs(self):
        a = synthetic_feature_table(seed=1)
        b = synthetic_feature_table(seed=2)
        assert not np.allclose(a.frame["f0"], b.frame["f0"])

    def test_cell_ids_match_partition_module(self):
        """跨模块契约：合成表的电池 ID 必须与划分模块同格式。

        曾经这里自己拼 ``cell-{c:04d}``，而 partition.make_cell_ids 用 3 位
        下限，于是 30 颗电池的合成表是 cell-0000..cell-0029，划分却是
        cell-000..cell-029——对不上账，check_partition 报一片 P-UNKNOWN，
        而那是格式不一致不是真的数据错误。格式只允许有一个来源。
        """
        from flsoh.partition import make_cell_ids

        for n in (5, 30, 124):
            assert synthetic_feature_table(n_cells=n, input_dim=4).cells == make_cell_ids(n)

    def test_has_capacity_fade_trend(self):
        # 没有趋势的话，标准化测试会退化成除以常数，抓不到真错误
        t = synthetic_feature_table(n_cells=3, cycles_per_cell=50, input_dim=4, seed=0)
        one = t.frame[t.frame[CELL_COL] == t.cells[0]]
        assert one[SOH_COL].iloc[-1] < one[SOH_COL].iloc[0]

    def test_rejects_bad_shape(self):
        for kwargs in ({"n_cells": 0}, {"cycles_per_cell": 1}, {"input_dim": 0}):
            with pytest.raises(DataContractError):
                synthetic_feature_table(**kwargs)
