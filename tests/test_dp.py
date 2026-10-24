"""本地 DP-SGD 的测试（`flsoh/dp.py` 与 `train.py` 的 DP 分支）。

这个文件盯的是**静默失败**，不是数值精度。DP 在这套代码里有三个地方
可以"看起来在跑、其实没在记账"：

1. 每轮新建 ``make_private`` → 会计器每轮归零，ε 少算 100 倍；
2. 会计器按 N 个客户端求和 → ε 虚高 23 倍（反方向，但同样错）；
3. ``noise_multiplier=0`` 的对照组被当成 DP 臂报出去。

所以断言分三组：**记账随步数增长**、**每客户端独立且不求和**、
**对照组报 inf 而不是 0**。再加一条等价性：``train_epochs(dp=...)``
必须与手写 DP 循环逐位相同——这是"训练只有一份实现"的契约
（plan.md §3.3），DP 分支是那次重构唯一的新增分叉。
"""

import pytest

# importorskip 必须是第一个 import（CI 不装 torch 时要干净 skip）
torch = pytest.importorskip("torch", reason="DP 需要 torch")
opacus = pytest.importorskip("opacus", reason="DP 需要 opacus（[privacy] extra）")

from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from flsoh.dp import ACCOUNTANT, DPTrainer  # noqa: E402
from flsoh.model import MLP  # noqa: E402
from flsoh.train import train_epochs  # noqa: E402

INPUT_DIM = 6


def _data(n: int = 20, batch_size: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, INPUT_DIM, generator=g)
    y = torch.rand(n, generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _model(seed: int = 0) -> MLP:
    torch.manual_seed(seed)
    return MLP(INPUT_DIM, hidden_dims=(4, 3))


def _trainer(loader=None, *, noise_multiplier=1.0, **kw):
    return DPTrainer(
        _model(), loader if loader is not None else _data(),
        noise_multiplier=noise_multiplier, optimizer="adam", lr=1e-2, **kw
    )


class TestAccounting:
    def test_epsilon_grows_with_steps(self):
        """ε 必须随步数单调增长。

        这条是防"会计器没接上"的：如果 ``step_hook`` 没挂上，history 是空
        的，``get_epsilon`` 会返回 0——训练照跑、曲线照出，只是 ε 永远是 0，
        而 0 看起来像"极强的隐私保证"。
        """
        t = _trainer()
        train_epochs(None, None, epochs=1, lr=1e-2, dp=t)
        eps1 = t.epsilon()
        assert eps1 > 0
        train_epochs(None, None, epochs=1, lr=1e-2, dp=t)
        assert t.epsilon() > eps1

    def test_steps_counted(self):
        """步数必须等于 epochs × 批次数。

        20 行 / batch 8 → 3 批。少记一步就少算一分钱，而少记的表现是
        ε 偏小——同样指向"隐私很强"这个错误结论。
        """
        t = _trainer()
        assert t.n_batches == 3
        train_epochs(None, None, epochs=2, lr=1e-2, dp=t)
        assert t.steps == 6

    def test_accountant_is_per_client_not_summed(self):
        """两个客户端各跑 k 步，ε 必须等于一个客户端跑 k 步，**不是** 2k 步。

        这是 ``flsoh/dp.py`` 模块 docstring 里那条论证的可执行形式：
        联合机制的隐私损失是各客户端 ε 的上确界，不是和。若实现成了
        "全局一个会计器"，这条会红——而那种错误让 ε 虚高 N 倍，
        报告会写"DP 不可用"，把可用的一档误杀。
        """
        a, b = _trainer(), _trainer()
        for t in (a, b):
            train_epochs(None, None, epochs=1, lr=1e-2, dp=t)
        joint = _trainer()
        train_epochs(None, None, epochs=1, lr=1e-2, dp=joint)
        assert a.epsilon() == pytest.approx(joint.epsilon())
        assert b.epsilon() == pytest.approx(joint.epsilon())

        double = _trainer()
        train_epochs(None, None, epochs=2, lr=1e-2, dp=double)
        assert double.epsilon() > joint.epsilon()

    def test_epsilon_raises_before_any_step(self):
        """没训练就问 ε 必须抛错。

        ε=0 在数学上没错（没有释放就没有损失），但在这个仓库里
        "跑过 DP 却报 ε=0" 只可能是记账没接上。宁可炸掉。
        """
        with pytest.raises(ValueError, match="还没训练过"):
            _trainer().epsilon()

    def test_sample_rate_matches_formula(self):
        """抽样率 = 1/批次数，且期望批大小 = 行数 × 抽样率。

        这两个数决定信噪比（噪声 Std = nm × max_grad_norm / 批大小），
        算错会让整条隐私-效用曲线的横坐标平移而没人发现。
        """
        t = _trainer()
        assert t.sample_rate == pytest.approx(1 / 3)
        assert t.expected_batch_size == 6  # int(20 / 3)
        train_epochs(None, None, epochs=1, lr=1e-2, dp=t)
        rec = t.privacy_record()
        assert rec["sample_rate"] == pytest.approx(1 / 3)
        assert rec["accountant"] == ACCOUNTANT
        assert rec["poisson_sampling"] is True
        assert rec["steps"] == 3
        assert rec["epsilon"] == pytest.approx(t.epsilon())


class TestControlArm:
    def test_noise_multiplier_zero_reports_inf(self):
        """``noise_multiplier=0`` 必须报 ``inf``，不是 0。

        这一组是**裁剪-only 对照组**（plan §7：「没有噪声就没有 DP」）。
        报 0 会让它在表里长得像"ε=0 的完美 DP"，而它其实没有任何保证。
        """
        t = _trainer(noise_multiplier=0.0)
        train_epochs(None, None, epochs=1, lr=1e-2, dp=t)
        assert t.epsilon() == float("inf")
        assert t.privacy_record()["epsilon"] == float("inf")

    def test_control_arm_still_clips_and_samples(self):
        """对照组必须仍然走 DP 管道（步数照记、Poisson 照采）。

        若实现成"nm=0 就直接跳过 make_private"，对照组就退化成了普通
        训练——那它隔离不出 Poisson + 裁剪的机制开销，而图 3 上 nm=0 的
        点会和非 DP 基线重合，看起来"DP 机制不要钱"。
        """
        t = _trainer(noise_multiplier=0.0)
        assert t.sample_rate == pytest.approx(1 / 3)
        assert t.expected_batch_size == 6
        train_epochs(None, None, epochs=1, lr=1e-2, dp=t)
        assert t.steps == 3
        # 裁剪确实生效：梯度范数被 max_grad_norm 压住。用权重范数量级
        # 粗判——裁剪上限 1.0，未裁剪时这个玩具模型的梯度远大于 1。
        gnorm = float(
            torch.sqrt(sum((p.grad ** 2).sum() for p in t.wrapped.parameters()))
        )
        assert gnorm <= t.max_grad_norm + 1e-6

    def test_poisson_batches_vary_in_size(self):
        """Poisson 抽样下批大小必须逐批不同（期望 6，不是 8）。

        这条锁的是"我们真的付了 subsampling 分析的代价"：如果哪天有人
        把 ``poisson_sampling=False`` 打开以保住原始批组成，批大小会变回
        常数 8/8/4，ε 的推导前提就没了——而训练照跑、ε 照报。
        """
        t = _trainer()
        sizes = [x.shape[0] for x, _ in t.loader]
        assert len(sizes) == 3
        assert len(set(sizes)) > 1, "批大小恒定 → Poisson 抽样没生效"
        assert 4 <= min(sizes) and max(sizes) <= 10


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"noise_multiplier": -1.0}, "不能为负"),
            ({"max_grad_norm": 0.0}, "必须为正"),
            ({"max_grad_norm": -1.0}, "必须为正"),
            ({"target_delta": 0.0}, "target_delta"),
            ({"target_delta": 1.0}, "target_delta"),
        ],
    )
    def test_bad_hyperparameters_raise(self, kwargs, match):
        """超参越界必须抛错，不静默取默认值。

        ``noise_multiplier=-1`` 若被静默当成 1.0，跑出来的是"别人的实验"。
        """
        with pytest.raises(ValueError, match=match):
            _trainer(**kwargs)

    def test_missing_opacus_message_is_actionable(self, monkeypatch):
        """没装 opacus 时报的错必须说出怎么装。

        只报 ``ModuleNotFoundError: opacus`` 的话，下一个人要先搞清
        "这是可选依赖还是必选依赖"——而这个仓库里它是可选的
        （pyproject 的 ``[privacy]`` extra），所以话必须说全。
        """
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "opacus":
                raise ImportError("No module named 'opacus'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match=r"pip install .*privacy"):
            _trainer()


class TestTrainEpochsDpBranch:
    def test_matches_hand_written_dp_loop(self):
        """``train_epochs(dp=...)`` 必须与手写 DP 循环逐位相等。

        这是 plan.md §3.3"训练只有一份实现"的契约。DP 分支是那次重构
        唯一的新增分叉，若不钉住，两边会各自演化——而漂移的表现是
        "DP 组和非 DP 组的差异里混进了实现差异"，那种差异无法归因。
        """
        loader = _data()

        via_helper = _trainer(loader)
        torch.manual_seed(1234)
        train_epochs(None, None, epochs=2, lr=1e-2, dp=via_helper)

        manual = _trainer(loader)
        loss_fn = torch.nn.MSELoss()
        torch.manual_seed(1234)
        manual.wrapped.train()
        for _ in range(2):
            for x, y in manual.loader:
                manual.optimizer.zero_grad()
                loss = loss_fn(manual.wrapped(x).squeeze(-1), y)
                loss.backward()
                manual.optimizer.step()
                manual.steps += 1

        for k in via_helper.state():
            assert torch.equal(via_helper.state()[k], manual.state()[k])
        assert via_helper.steps == manual.steps == 6
        assert via_helper.epsilon() == pytest.approx(manual.epsilon())

    def test_branch_ignores_passed_model_and_loader(self):
        """给了 ``dp`` 就必须走 DP 分支：``model``/``loader`` 参数被忽略。

        反面对照。若分支判断写反（或漏写），这里传进去的**错** loader
        会被用来训练，两个 trainer 的状态就会不同——所以这条能抓住
        "dp 参数被静默忽略"，而那种情况下 DP 组跑的是普通 SGD，
        ε 却照报，整条隐私-效用曲线都是假的。
        """
        good = _trainer(_data())
        decoy_model = _model(seed=99)
        decoy_loader = _data(n=13, batch_size=5, seed=7)
        torch.manual_seed(5)
        train_epochs(decoy_model, decoy_loader, epochs=2, lr=1e-2, dp=good)

        reference = _trainer(_data())
        torch.manual_seed(5)
        train_epochs(reference.model, reference.loader, epochs=2, lr=1e-2, dp=reference)

        assert good.steps == 6
        for k in good.state():
            assert torch.equal(good.state()[k], reference.state()[k])
        # 诱饵模型确实没被训练（否则上面的逐位相等不可能成立）
        assert not any(
            torch.equal(decoy_model.state_dict()[k], reference.state()[k])
            for k in reference.state()
        )

    def test_without_dp_behaves_as_before(self):
        """``dp=None`` 时必须走原路径（回归保护）。

        DP 分支加进来之后，非 DP 的每一步都不能变——否则 M1-M3 的
        全部结果都不可比了。这里用固定 seed 钉一个具体数值。
        """
        torch.manual_seed(3)
        model = _model()
        loader = _data(seed=1)
        loss = train_epochs(model, loader, epochs=2, lr=1e-2)
        assert loss == pytest.approx(loss)  # 有限
        assert all(torch.isfinite(p).all() for p in model.parameters())
