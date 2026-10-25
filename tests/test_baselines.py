"""``flsoh.baselines`` 的测试。

这一组测试的重点不是「岭回归算得对」（那是 numpy 的事），而是**它不能以
看不出来的方式变好**：标准化漏了测试集信息、常数列变成 nan、形状那一路
悄悄用上了幅度——三种都会让阶梯表给出一个「模型很好」的假象。
"""

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="baselines 需要 numpy")

from flsoh.baselines import (  # noqa: E402
    amplitude,
    feature_ladder,
    ladder_note,
    ladder_verdict,
    ridge_predict,
    scalar_summaries,
)


class TestAmplitude:
    def test_is_max_absolute_value_per_row(self):
        X = np.array([[1.0, -3.0, 2.0], [0.5, 0.1, -0.2]])
        got = amplitude(X)
        assert got.shape == (2, 1)
        assert got.ravel().tolist() == [3.0, 0.5]

    def test_sign_does_not_matter(self):
        """ΔQ 是「相对参考循环的变化」，整体为负是正常的（容量衰减）。"""
        assert amplitude(np.array([[-2.0, 1.0]]))[0, 0] == 2.0

    def test_rejects_one_dimensional_input(self):
        with pytest.raises(ValueError, match="二维"):
            amplitude(np.array([1.0, 2.0, 3.0]))


class TestRidgePredict:
    def test_alpha_zero_recovers_an_exact_linear_relation(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 3))
        y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5
        pred = ridge_predict(X, y, X, alpha=0.0)
        assert np.abs(pred - y).max() < 1e-8

    def test_the_default_penalty_shrinks_but_keeps_the_relation(self):
        """默认 ``alpha=1.0`` 是**有**收缩的，所以不会逐位复原。

        这一条钉住「收缩有多大」：相对 y 的标准差应在 5% 以内。如果哪天
        有人把默认 alpha 调到 100，这里会失败，而失败的样子正是阶梯表里
        的「全部特征」一格莫名变差。
        """
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 3))
        y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5
        pred = ridge_predict(X, y, X)
        assert np.abs(pred - y).max() / y.std() < 0.05

    def test_larger_alpha_shrinks_further_from_the_unpenalised_fit(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(80, 5))
        y = X @ np.array([1.0, -2.0, 0.5, 0.0, 3.0])
        free = ridge_predict(X, y, X, alpha=0.0)
        weak = ridge_predict(X, y, X, alpha=1.0)
        strong = ridge_predict(X, y, X, alpha=1000.0)
        err = lambda p: np.abs(p - y).mean()  # noqa: E731
        assert err(free) < err(weak) < err(strong)

    def test_constant_column_does_not_produce_nan(self):
        """常数列除以 0 会让整个解变成 nan，而 nan 的 MAE 也是 nan。

        模拟数据里整列相同是常态（网格两端裁剪后没有变化），所以这一条
        不是边界情况，是主路径。
        """
        X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        pred = ridge_predict(X, y, X)
        assert np.isfinite(pred).all()

    def test_prediction_for_a_row_does_not_depend_on_other_test_rows(self):
        """标准化统计量只能来自训练集。

        如果实现改成「训练集 + 测试集一起算 mean/std」，同一行的预测会随
        **别的测试行**变化。这个性质用肉眼看不出来，但它是信息泄漏的定义。
        """
        rng = np.random.default_rng(1)
        X_train = rng.normal(size=(50, 4))
        y_train = rng.normal(size=50)
        X_test = rng.normal(size=(10, 4))

        alone = ridge_predict(X_train, y_train, X_test[[0]])
        together = ridge_predict(X_train, y_train, X_test)
        assert alone[0] == pytest.approx(together[0], abs=1e-12)

    def test_rejects_mismatched_feature_counts(self):
        with pytest.raises(ValueError, match="维数不同"):
            ridge_predict(np.zeros((4, 2)), np.zeros(4), np.zeros((3, 3)))

    def test_rejects_row_count_mismatch(self):
        with pytest.raises(ValueError, match="行"):
            ridge_predict(np.zeros((4, 2)), np.zeros(3), np.zeros((2, 2)))


def _amplitude_only_problem(n_cells=12, n_cycles=10, seed=0):
    """造一份「标签完全由幅度决定、形状恒定」的特征表。

    这是模拟数据的真实结构：``Q(c)`` 只缩放整条曲线。形状那一格因此没有
    任何信号，只该拟合到噪声。
    """
    rng = np.random.default_rng(seed)
    base = np.linspace(0.0, 1.0, 40) ** 0.7
    rows, labels, cells = [], [], []
    for c in range(n_cells):
        scale_c = 0.8 + 0.4 * c / n_cells  # 每颗电池的容量因子
        for k in range(n_cycles):
            amp = scale_c * (1.0 - 0.02 * k)  # 幅度随循环衰减
            rows.append(amp * base)
            labels.append(amp)  # 标签就是幅度
            cells.append(f"c{c:02d}")
    return np.array(rows), np.array(labels), np.array(cells)


class TestFeatureLadder:
    def _masks(self, cells):
        train = cells != "c00"
        return train, ~train

    def test_amplitude_wins_when_the_label_is_the_amplitude(self):
        X, y, cells = _amplitude_only_problem()
        train, test = self._masks(cells)
        ladder = feature_ladder(X, y, train, test)
        assert ladder["amplitude"] < ladder["constant"] / 10
        # 形状里没有标签信息 → 只该拟合到噪声，比水平线还差
        assert ladder["shape_only"] > ladder["constant"]

    def test_counts_are_reported(self):
        X, y, cells = _amplitude_only_problem()
        train, test = self._masks(cells)
        ladder = feature_ladder(X, y, train, test)
        assert ladder["n_train"] == int(train.sum())
        assert ladder["n_test"] == int(test.sum())
        assert ladder["n_features"] == X.shape[1]

    def test_constant_baseline_matches_the_definition(self):
        X, y, cells = _amplitude_only_problem()
        train, test = self._masks(cells)
        ladder = feature_ladder(X, y, train, test)
        expected = np.abs(y[test] - y[train].mean()).mean()
        assert ladder["constant"] == pytest.approx(expected)

    def test_overlapping_masks_are_refused(self):
        """同一行既训练又测试，MAE 会好得毫无道理。宁可报错也不给数。"""
        X, y, cells = _amplitude_only_problem()
        train = np.ones(len(y), dtype=bool)
        with pytest.raises(ValueError, match="重叠"):
            feature_ladder(X, y, train, train)

    def test_empty_split_is_refused(self):
        X, y, cells = _amplitude_only_problem()
        train, test = self._masks(cells)
        with pytest.raises(ValueError, match="为空"):
            feature_ladder(X, y, np.zeros(len(y), dtype=bool), test)

    def test_length_mismatch_is_refused(self):
        X, y, cells = _amplitude_only_problem()
        train, test = self._masks(cells)
        with pytest.raises(ValueError, match="掩码长度"):
            feature_ladder(X, y, train[:-1], test)


class TestLadderNote:
    def test_worse_than_amplitude_says_one_dimension_is_enough(self):
        ladder = {"amplitude": 0.0043, "all_features": 0.0057}
        note = ladder_note(ladder, neural_mae=0.0229)
        assert "1 维线性模型就够了" in note
        assert "×5.33" in note

    def test_better_than_amplitude_credits_the_extra_dimensions(self):
        ladder = {"amplitude": 0.0100, "all_features": 0.0040}
        note = ladder_note(ladder, neural_mae=0.0050)
        assert "幅度之外的信号" in note
        assert "1 维线性模型就够了" not in note

    def test_without_neural_mae_it_only_states_the_ladder(self):
        ladder = {"amplitude": 0.0043, "all_features": 0.0057}
        note = ladder_note(ladder)
        assert "最强平凡基线" in note
        assert "模型" not in note


class TestE2EDefaults:
    """``run_e2e.py`` 的默认训练量必须**复现已提交的产物**。

    这条闸门的来历：默认曾经是 ``local_epochs=2``（共 120 epoch），跑出来
    MAE 0.05623，**比常数基线 0.02638 还差**，直接判 FAIL。而
    ``results/e2e/e2e__sim__seed0.json`` 里的 0.02289 是 ``local_epochs=5``
    （共 300）跑的。于是「照着文档跑一条命令」和「仓库里那个数」对不上，
    对不上的方式是**第一次跑就红灯**。

    改成 5 的唯一理由是它与归档一致——不是因为它更好。同一份数据扫
    ``local_epochs`` 是 2 → 0.0562、3 → 0.0229、4 → 0.0620、5 → 0.0229、
    8 → 0.0098，非单调，所以「挑一个能过的设置」既是挑最好的一次，也不成立。
    """

    def test_defaults_reproduce_the_archived_run(self):
        pytest.importorskip("pyarrow", reason="run_e2e 需要 pyarrow")
        from run_e2e import DEFAULT_LOCAL_EPOCHS, DEFAULT_ROUNDS

        archive = (
            Path(__file__).resolve().parent.parent
            / "results" / "e2e" / "e2e__sim__seed0.json"
        )
        payload = json.loads(archive.read_text(encoding="utf-8"))
        assert DEFAULT_ROUNDS * DEFAULT_LOCAL_EPOCHS == payload["epochs"], (
            f"默认 {DEFAULT_ROUNDS} 轮 × {DEFAULT_LOCAL_EPOCHS} 本地 epoch = "
            f"{DEFAULT_ROUNDS * DEFAULT_LOCAL_EPOCHS}，但归档产物是 "
            f"{payload['epochs']}。默认值必须复现已提交的产物，"
            "否则「跑默认命令」会得到另一个数（还可能是红的）"
        )

    def test_archived_run_beats_the_constant_baseline(self):
        """归档值必须真的跑赢常数基线，否则它不该被当结果留着。"""
        archive = (
            Path(__file__).resolve().parent.parent
            / "results" / "e2e" / "e2e__sim__seed0.json"
        )
        payload = json.loads(archive.read_text(encoding="utf-8"))
        assert payload["final_mae"] < payload["baseline_mae"]


class TestScalarSummaries:
    def test_every_summary_is_a_column(self):
        X = np.random.default_rng(0).normal(size=(17, 30))
        for name, col in scalar_summaries(X).items():
            assert col.shape == (17, 1), name
            assert np.isfinite(col).all(), name

    def test_max_summary_agrees_with_amplitude(self):
        X = np.random.default_rng(1).normal(size=(11, 9))
        assert np.allclose(scalar_summaries(X)["max|ΔQ|"], amplitude(X))

    def test_magnitude_summaries_are_sign_invariant(self):
        """ΔQ 整体为负是正常的（容量衰减），**幅度类**摘要不该依赖符号。

        「中点电压处的值」是**带符号**的单个采样点，不在这一条里——它不是
        幅度摘要，岭回归可以自己翻符号。第一版把六个摘要一起断言，当场被
        这一格否掉：测试的前提错了，不是代码错了。
        """
        rng = np.random.default_rng(2)
        X = rng.normal(size=(9, 12))
        pos = scalar_summaries(X)
        neg = scalar_summaries(-X)
        magnitudes = [k for k in pos if k != "中点电压处的值"]
        assert len(magnitudes) == 5
        for name in magnitudes:
            assert np.allclose(pos[name], neg[name]), name

    def test_signed_summary_is_not_claimed_to_be_a_magnitude(self):
        X = np.array([[1.0, -2.0, 3.0]])
        assert scalar_summaries(X)["中点电压处的值"][0, 0] == -2.0

    def test_rejects_one_dimensional_input(self):
        with pytest.raises(ValueError, match="二维"):
            scalar_summaries(np.arange(5.0))


class TestLadderVerdict:
    """四种情形必须分开判。

    上一版把「形状不如常数基线」直接读成「幅度是唯一载体」，而真实数据上
    幅度也比常数基线差——一句话把反直觉的发现读成了「符合预期」。
    """

    def _ladder(self, *, constant, amplitude, all_features, shape_only):
        return {"constant": constant, "amplitude": amplitude,
                "all_features": all_features, "shape_only": shape_only}

    def test_both_single_views_carry_information(self):
        v = ladder_verdict(self._ladder(constant=1.0, amplitude=0.5,
                                        all_features=0.3, shape_only=0.8))
        assert "各自单独就带信息" in v

    def test_simulated_case_amplitude_only(self):
        """模拟数据：幅度好、形状是噪声（0.0721 > 0.0264）。"""
        v = ladder_verdict(self._ladder(constant=0.0264, amplitude=0.0035,
                                        all_features=0.0061, shape_only=0.0721))
        assert "信息在幅度里，形状是噪声" in v

    def test_shape_only_case(self):
        v = ladder_verdict(self._ladder(constant=1.0, amplitude=1.2,
                                        all_features=0.9, shape_only=0.5))
        assert "信息在形状里" in v

    def test_real_data_case_neither_view_alone(self):
        """真实 MATR：幅度 1.174、形状 2.099，都不如常数；全量 0.361。

        这一格最容易被读错——两个单视图都失败，但特征其实很有用。
        """
        v = ladder_verdict(self._ladder(constant=0.00179, amplitude=0.00210,
                                        all_features=0.00065, shape_only=0.00376))
        assert "任何一个一维摘要都不够" in v
        assert "组合" in v
        assert "3.23" in v
        assert "唯一载体" not in v

    def test_no_signal_at_all(self):
        v = ladder_verdict(self._ladder(constant=1.0, amplitude=1.1,
                                        all_features=1.2, shape_only=1.3))
        assert "没有可用信号" in v
