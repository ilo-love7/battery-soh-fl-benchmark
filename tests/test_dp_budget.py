"""DP 预算检查器的测试（`scripts/check_dp_budget.py`）。

为什么值得单独测：**ε 的横坐标错了，整条隐私-效用曲线就错了**，而曲线
画出来照样是一条漂亮的下降线。这里钉两件事：

1. 批次数必须把最后那个不满的批算进来（80 行 / batch 64 → **2** 批，
   不是 1）。漏掉它会把抽样率翻倍、ε 直接算小一半。
2. ε 必须随步数增长，且与手算的 RDP 对得上。

第二条尤其重要：如果哪天有人把会计器换成 PRV/GDP 而没改这里，曲线上的
ε 会整体平移，而没有任何东西会报错。
"""

from __future__ import annotations

from pathlib import Path

import pytest

# importorskip 必须是第一个 import（CI 不装 torch/opacus 时要干净 skip）
pytest.importorskip("torch", reason="DP 预算需要 opacus，而 opacus 需要 torch")
pytest.importorskip("opacus", reason="DP 预算需要 opacus（[privacy] extra）")

import check_dp_budget as cdb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class TestBatchesPerEpoch:
    def test_partial_last_batch_counts(self):
        """80 行 / batch 64 → 2 批（64 + 16）。

        DataLoader 默认 ``drop_last=False``，最后那个 16 行的小批会真的
        参与一次 ``optimizer.step()``，也就真的花一分钱隐私预算。漏数它
        会把批次数当成 1：抽样率从 0.5 变 1.0，ε 的算法整个换了一套，
        而训练日志一行都不会变。
        """
        assert cdb._batches_per_epoch(80, 64) == 2
        assert cdb._batches_per_epoch(20, 8) == 3      # 8 + 8 + 4
        assert cdb._batches_per_epoch(64, 64) == 1     # 正好一批
        assert cdb._batches_per_epoch(32, 64) == 1     # 不足一批仍算一批

    def test_rejects_non_positive_rows(self):
        with pytest.raises(ValueError, match="必须为正"):
            cdb._batches_per_epoch(0, 64)


class TestEpsilonFor:
    def test_matches_hand_rolled_accountant(self):
        """逐步记账与「一条 num_steps 记录」必须给出同一个 ε。

        RDP 对同 (σ, q) 的多步是可加的，两种记账法在数学上等价。这条
        断言的是实现没在哪一条路上少记一步。
        """
        from opacus.accountants import RDPAccountant

        eps_step = cdb.epsilon_for(1.0, 0.5, 20, 1e-5)
        acc = RDPAccountant()
        acc.step(noise_multiplier=1.0, sample_rate=0.5)
        for _ in range(19):
            acc.step(noise_multiplier=1.0, sample_rate=0.5)
        assert eps_step == pytest.approx(acc.get_epsilon(delta=1e-5))

    def test_grows_with_steps(self):
        a = cdb.epsilon_for(1.0, 0.5, 10, 1e-5)
        b = cdb.epsilon_for(1.0, 0.5, 100, 1e-5)
        assert b > a > 0

    def test_falls_with_noise_multiplier(self):
        assert cdb.epsilon_for(8.0, 0.5, 600, 1e-5) < cdb.epsilon_for(
            1.0, 0.5, 600, 1e-5
        )

    def test_mvp_schedule_numbers_are_pinned(self):
        """本仓库真实协议（q=0.5、600 步、δ=1e-5）下的 ε 必须逐个对上。

        这些数来自 `check_dp_budget.py --config configs/dp.toml` 的实际
        输出，也是 plan §13.4 预算表的来源。它们变了而没人注意，就等于
        整条隐私-效用曲线的横坐标被悄悄平移。
        """
        expected = {1.0: 152.31, 2.0: 49.11, 4.0: 18.65, 8.0: 7.84}
        for nm, eps in expected.items():
            got = cdb.epsilon_for(nm, 0.5, 600, 1e-5)
            assert got == pytest.approx(eps, rel=1e-3), (nm, got, eps)


class TestCli:
    def test_disabled_privacy_is_a_no_op(self, capsys):
        code = cdb.main(["--config", str(ROOT / "configs" / "mvp.toml")])
        assert code == 0
        assert "privacy.enabled=false" in capsys.readouterr().out

    def test_dp_config_reports_the_budget_table(self, capsys):
        code = cdb.main(["--config", str(ROOT / "configs" / "dp.toml")])
        out = capsys.readouterr().out
        assert code == 0
        assert "每客户端 80 行" in out
        assert "抽样率 q=0.5" in out
        assert "步数 = 100 轮 × 3 本地 epoch × 2 批 = 600" in out
        # 唯一达到 ε ≤ 8 的一档必须被点出来
        assert "达到 ε ≤ 8 的扫描点：[8]" in out

    def test_unreachable_budget_fails(self, tmp_path, capsys):
        """三档全都 ε>8 时必须退出码 1——不拿 nm=1 冒充可用点。

        退出码 1 的含义是「这份协议的 DP 判据不可用」，不是「脚本坏了」。
        plan §12 R5：DP 下学不动就是结论之一，如实报。
        """
        cfg = tmp_path / "tight.toml"
        cfg.write_text(
            (ROOT / "configs" / "dp.toml").read_text(encoding="utf-8").replace(
                "noise_multiplier = [0, 1, 2, 4, 8]",
                "noise_multiplier = [0, 0.5, 1.0, 2.0]",
            ),
            encoding="utf-8",
        )
        code = cdb.main(["--config", str(cfg)])
        out = capsys.readouterr().out
        assert code == 1
        assert "没有任何正的扫描点达到" in out
