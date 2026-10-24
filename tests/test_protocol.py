"""protocol 模块测试：每条校验规则都能报出来，且合法配置不误报。"""

import copy
from pathlib import Path

import pytest

from flsoh.protocol import (
    LEVEL_ERROR,
    LEVEL_WARN,
    ProtocolError,
    check_app_config_agreement,
    expected_clients_for,
    has_errors,
    load_config,
    validate_config,
)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def base_config():
    """一份最小合法配置。每个用例改一处，确认只有预期的那条报警。"""
    return {
        "meta": {"name": "unit"},
        "data": {
            "dataset": "mit-stanford",
            "n_cells": 124,
            "partition_atom": "cell",
        },
        "partitions": {
            "test_cells": 24,
            "split_seed": 0,
            "cells_per_client": [5],
            "remainder": "spread",
            "schemes": ["iid", "dirichlet"],
            "dirichlet_alpha": [0.5],
        },
        "strategies": {
            "include": ["centralized", "fedavg", "fedprox"],
            "proximal_mu": [0.0, 0.01, 0.1],
        },
        "train": {
            "num_rounds": 100,
            "local_epochs": 1,
            "learning_rate": 0.001,
            "batch_size": 64,
            "fraction_fit": 1.0,
        },
        "seeds": {"values": [0, 1, 2], "min_required": 3},
        "privacy": {
            "enabled": False,
            "noise_multiplier": [],
            "max_grad_norm": 1.0,
            "target_delta": 1e-5,
        },
        "acceptance": {
            "iid_degradation_max": 0.05,
            "non_iid_degradation_min": 0.10,
            "fedprox_improvement_min": 0.05,
            "seed_cv_max": 0.10,
            "dp_degradation_max": 0.30,
            "dp_epsilon_max": 8.0,
        },
    }


def codes(issues, level=None):
    return [c for lv, c, _ in issues if level is None or lv == level]


def mutate(path, value):
    cfg = base_config()
    node = cfg
    keys = path.split(".")
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    return cfg


class TestValidConfig:
    def test_base_config_is_clean(self):
        assert validate_config(base_config()) == []

    def test_mvp_toml_has_no_errors(self):
        cfg = load_config(CONFIG_DIR / "mvp.toml")
        assert not has_errors(validate_config(cfg))

    def test_full_toml_has_no_errors(self):
        cfg = load_config(CONFIG_DIR / "full.toml")
        assert not has_errors(validate_config(cfg))

    def test_full_toml_emits_nonbuiltin_warning(self):
        # fedbn 写进了 include，就该提醒它不是内置策略——这是有意的 WARN，不是 ERROR
        cfg = load_config(CONFIG_DIR / "full.toml")
        assert "C-NONBUILTIN" in codes(validate_config(cfg), LEVEL_WARN)


class TestAtom:
    def test_rejects_missing_atom(self):
        cfg = base_config()
        del cfg["data"]["partition_atom"]
        assert "C-ATOM" in codes(validate_config(cfg), LEVEL_ERROR)

    def test_rejects_cycle_atom(self):
        issues = validate_config(mutate("data.partition_atom", "cycle"))
        assert "C-ATOM" in codes(issues, LEVEL_ERROR)

    def test_rejects_missing_dataset(self):
        issues = validate_config(mutate("data.dataset", ""))
        assert "C-DATASET" in codes(issues, LEVEL_ERROR)


class TestPartitions:
    def test_requires_split_seed(self):
        cfg = base_config()
        del cfg["partitions"]["split_seed"]
        assert "C-SPLIT-SEED" in codes(validate_config(cfg), LEVEL_ERROR)

    def test_rejects_holdout_larger_than_dataset(self):
        issues = validate_config(mutate("partitions.test_cells", 124))
        assert "C-TEST" in codes(issues, LEVEL_ERROR)

    def test_rejects_unknown_remainder(self):
        issues = validate_config(mutate("partitions.remainder", "keep"))
        assert "C-REMAINDER" in codes(issues, LEVEL_ERROR)

    def test_rejects_empty_cells_per_client(self):
        issues = validate_config(mutate("partitions.cells_per_client", []))
        assert "C-CPC" in codes(issues, LEVEL_ERROR)

    def test_rejects_non_integer_k(self):
        issues = validate_config(mutate("partitions.cells_per_client", [2.5]))
        assert "C-CPC" in codes(issues, LEVEL_ERROR)

    def test_rejects_k_larger_than_available(self):
        issues = validate_config(mutate("partitions.cells_per_client", [500]))
        assert "C-DIVIDE" in codes(issues, LEVEL_ERROR)

    def test_drop_remainder_warns_about_lost_cells(self):
        cfg = mutate("partitions.remainder", "drop")
        cfg["partitions"]["cells_per_client"] = [7]  # 100 % 7 = 2
        issues = validate_config(cfg)
        assert "C-DIVIDE" in codes(issues, LEVEL_WARN)
        assert not has_errors(issues)

    def test_requires_iid_control(self):
        issues = validate_config(mutate("partitions.schemes", ["dirichlet"]))
        assert "C-SCHEME" in codes(issues, LEVEL_ERROR)

    def test_rejects_unknown_scheme(self):
        issues = validate_config(mutate("partitions.schemes", ["iid", "shard"]))
        assert "C-SCHEME" in codes(issues, LEVEL_ERROR)

    def test_requires_alpha_when_dirichlet(self):
        issues = validate_config(mutate("partitions.dirichlet_alpha", []))
        assert "C-ALPHA" in codes(issues, LEVEL_ERROR)

    def test_rejects_non_positive_alpha(self):
        issues = validate_config(mutate("partitions.dirichlet_alpha", [0.0]))
        assert "C-ALPHA" in codes(issues, LEVEL_ERROR)


class TestStrategies:
    def test_requires_centralized_baseline(self):
        issues = validate_config(mutate("strategies.include", ["fedavg", "fedprox"]))
        assert "C-STRATEGY" in codes(issues, LEVEL_ERROR)

    def test_rejects_unknown_strategy(self):
        issues = validate_config(
            mutate("strategies.include", ["centralized", "fedavg", "fedmagic"])
        )
        assert "C-STRATEGY" in codes(issues, LEVEL_ERROR)

    def test_warns_when_all_mu_are_zero(self):
        issues = validate_config(mutate("strategies.proximal_mu", [0.0]))
        assert "C-PROX" in codes(issues, LEVEL_WARN)
        assert not has_errors(issues)

    def test_requires_mu_grid_when_fedprox(self):
        issues = validate_config(mutate("strategies.proximal_mu", []))
        assert "C-PROX" in codes(issues, LEVEL_ERROR)

    def test_warns_on_non_builtin_strategy(self):
        issues = validate_config(
            mutate("strategies.include", ["centralized", "fedavg", "scaffold"])
        )
        assert "C-NONBUILTIN" in codes(issues, LEVEL_WARN)


class TestTrain:
    def test_rejects_zero_rounds(self):
        issues = validate_config(mutate("train.num_rounds", 0))
        assert "C-TRAIN" in codes(issues, LEVEL_ERROR)

    def test_rejects_fraction_fit_above_one(self):
        issues = validate_config(mutate("train.fraction_fit", 1.5))
        assert "C-TRAIN" in codes(issues, LEVEL_ERROR)

    def test_warns_on_unbalanced_training_budget(self):
        """判据是「联邦总本地 epoch 与基线对不对等」，不是「le 大不大」。

        同一条 le=10，在每客户端 80 行数据下是必要的（le=2 欠拟合到 0.0222），
        在每客户端上万样本下才是漂移风险。所以只有比值才有意义。
        """
        cfg = mutate("train.local_epochs", 5)
        cfg["train"]["baseline_epochs"] = 300   # 500 / 300 = 1.67
        issues = validate_config(cfg)
        assert "C-TRAIN" in codes(issues, LEVEL_WARN)
        assert not has_errors(issues)

    def test_balanced_budget_is_not_warned(self):
        """等训练量必须放行，否则「按规矩对齐」反而会被报警。"""
        cfg = mutate("train.local_epochs", 3)
        cfg["train"]["baseline_epochs"] = 300   # 300 / 300 = 1.00
        issues = validate_config(cfg)
        assert "C-TRAIN" not in codes(issues, LEVEL_WARN)

    def test_missing_baseline_epochs_skips_budget_check(self):
        """baseline_epochs 是可选的：缺了就跳过，不能凭空造一个分母。"""
        cfg = mutate("train.local_epochs", 50)
        assert "baseline_epochs" not in cfg["train"]
        issues = validate_config(cfg)
        assert "C-TRAIN" not in codes(issues, LEVEL_WARN)


class TestSeeds:
    def test_rejects_two_seeds(self):
        cfg = mutate("seeds.values", [0, 1])
        issues = validate_config(cfg)
        assert "C-SEEDS" in codes(issues, LEVEL_ERROR)

    def test_rejects_low_min_required(self):
        cfg = base_config()
        cfg["seeds"]["min_required"] = 1
        assert "C-SEEDS" in codes(validate_config(cfg), LEVEL_ERROR)

    def test_rejects_duplicate_seeds(self):
        issues = validate_config(mutate("seeds.values", [0, 0, 1]))
        assert "C-SEEDS" in codes(issues, LEVEL_ERROR)

    def test_rejects_missing_enabled_flag(self):
        cfg = base_config()
        del cfg["privacy"]["enabled"]
        assert "C-PRIVACY" in codes(validate_config(cfg), LEVEL_ERROR)


class TestPrivacy:
    def _dp(self, **overrides):
        cfg = base_config()
        cfg["privacy"]["enabled"] = True
        cfg["privacy"]["noise_multiplier"] = [0.5, 1.0, 2.0]
        cfg["privacy"].update(overrides)
        return cfg

    def test_three_noise_points_accepted(self):
        assert validate_config(self._dp()) == []

    def test_rejects_two_noise_points(self):
        issues = validate_config(self._dp(noise_multiplier=[0.5, 1.0]))
        assert "C-PRIVACY" in codes(issues, LEVEL_ERROR)

    def test_rejects_non_positive_clipping_norm(self):
        issues = validate_config(self._dp(max_grad_norm=0.0))
        assert "C-PRIVACY" in codes(issues, LEVEL_ERROR)

    def test_rejects_bad_delta(self):
        issues = validate_config(self._dp(target_delta=1.0))
        assert "C-PRIVACY" in codes(issues, LEVEL_ERROR)

    def test_disabled_privacy_ignores_empty_noise_list(self):
        cfg = base_config()
        cfg["privacy"]["noise_multiplier"] = []
        assert validate_config(cfg) == []

    def test_zero_noise_is_a_control_arm_not_a_point(self):
        """nm=0 是对照组，不计入「≥3 个扫描点」。

        把它算进去的话，`[0, 1, 2]` 会被判成三点通过，而隐私-效用曲线
        实际只有两个 DP 点——表里写着 3，图画出来是 2。
        """
        issues = validate_config(self._dp(noise_multiplier=[0, 1.0, 2.0]))
        assert "C-PRIVACY" in codes(issues, LEVEL_ERROR)

    def test_zero_plus_three_positive_is_accepted_with_a_warning(self):
        """`[0, 1, 2, 4]` 应当通过，但要警告 0 不是 DP 臂。

        plan §7：「没有噪声就没有 DP」。对照组必须能跑，但报告里不能
        把它算作隐私保证的一档。
        """
        issues = validate_config(self._dp(noise_multiplier=[0, 1.0, 2.0, 4.0]))
        assert not has_errors(issues)
        assert "C-PRIVACY" in codes(issues, LEVEL_WARN)
        assert any("对照组" in m for _, _, m in issues)

    def test_all_zero_is_rejected(self):
        issues = validate_config(self._dp(noise_multiplier=[0, 0]))
        assert "C-PRIVACY" in codes(issues, LEVEL_ERROR)

    def test_negative_noise_is_rejected(self):
        issues = validate_config(self._dp(noise_multiplier=[-1.0, 1.0, 2.0]))
        assert "C-PRIVACY" in codes(issues, LEVEL_ERROR)


class TestAcceptance:
    def test_rejects_missing_threshold(self):
        cfg = base_config()
        del cfg["acceptance"]["seed_cv_max"]
        assert "C-ACCEPT" in codes(validate_config(cfg), LEVEL_ERROR)

    def test_rejects_threshold_outside_unit_interval(self):
        issues = validate_config(mutate("acceptance.iid_degradation_max", 5.0))
        assert "C-ACCEPT" in codes(issues, LEVEL_ERROR)

    def test_rejects_zero_threshold(self):
        # 阈值 0 意味着"任何劣化都不可接受"，实际上等于没设判据
        issues = validate_config(mutate("acceptance.fedprox_improvement_min", 0.0))
        assert "C-ACCEPT" in codes(issues, LEVEL_ERROR)

    def test_epsilon_is_not_a_fraction(self):
        # ε 是预算值不是比例，>1 合法
        issues = validate_config(mutate("acceptance.dp_epsilon_max", 50.0))
        assert "C-ACCEPT" not in codes(issues, LEVEL_ERROR)


class TestUnknownSection:
    def test_warns_on_unrecognised_section(self):
        cfg = base_config()
        cfg["experiment"] = {"foo": 1}
        issues = validate_config(cfg)
        assert "C-UNKNOWN" in codes(issues, LEVEL_WARN)
        assert not has_errors(issues)


class TestLoadConfig:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ProtocolError):
            load_config(tmp_path / "nope.toml")

    def test_broken_toml_raises(self, tmp_path):
        bad = tmp_path / "bad.toml"
        bad.write_text("[data\nn_cells = 1\n", encoding="utf-8")
        with pytest.raises(ProtocolError):
            load_config(bad)

    def test_reads_real_config(self):
        cfg = load_config(CONFIG_DIR / "mvp.toml")
        assert cfg["data"]["partition_atom"] == "cell"
        assert cfg["meta"]["name"] == "mvp"


class TestExpectedClientsFor:
    def test_mvp(self):
        # 135 颗、留 20 颗测试 → 115 颗进客户端；k=5 → 23。
        # 这个数改过两次（25 → 20 → 23），每次都是因为上游的颗数变了。
        assert expected_clients_for(load_config(CONFIG_DIR / "mvp.toml"), 5) == 23

    def test_full_two_k_values(self):
        cfg = load_config(CONFIG_DIR / "full.toml")
        assert expected_clients_for(cfg, 2) == 58
        assert expected_clients_for(cfg, 5) == 23


class TestConfigsAreNotCopies:
    def test_mvp_and_full_differ_in_seed_budget(self):
        # 两份配置若只在注释上不同，说明矩阵根本没扩，图 2/图 3 会没数据
        mvp = load_config(CONFIG_DIR / "mvp.toml")
        full = load_config(CONFIG_DIR / "full.toml")
        assert full["partitions"]["dirichlet_alpha"] != mvp["partitions"]["dirichlet_alpha"]
        assert full["privacy"]["enabled"] is True
        assert mvp["privacy"]["enabled"] is False


class TestCopySafety:
    def test_base_config_returns_fresh_object(self):
        a = base_config()
        b = base_config()
        a["data"]["n_cells"] = 1
        assert b["data"]["n_cells"] == 124

    def test_mutate_does_not_touch_original(self):
        original = copy.deepcopy(base_config())
        mutate("data.n_cells", 1)
        assert base_config() == original


class TestAppConfigAgreement:
    """pyproject 的 run_config 与协议配置必须一致。

    两边分开维护、改一份忘另一份时**谁都不报错**——只是官方运行时与本地
    循环跑的不是同一个实验。2026-10-24 实际发生过（pyproject 停在 2 轮 ×
    2 ep，configs/mvp.toml 已是 100 轮 × 10 ep），所以要有这条检查。
    """

    def _pyproject(self, tmp_path, **app_cfg):
        body = "\n".join(f"{k} = {v!r}" if isinstance(v, str) else f"{k} = {v}"
                         for k, v in app_cfg.items())
        p = tmp_path / "pyproject.toml"
        p.write_text("[tool.flwr.app]\nconfig = {}\n" if not app_cfg else
                     f"[tool.flwr.app.config]\n{body}\n", encoding="utf-8")
        return p

    def _cfg(self):
        cfg = base_config()
        cfg["train"]["baseline_epochs"] = 300
        return cfg

    def test_matching_values_pass(self, tmp_path):
        p = self._pyproject(tmp_path, **{
            "num-server-rounds": 100,
            "local-epochs": 1,
            "learning-rate": 0.001,
            "batch-size": 64,
        })
        issues = check_app_config_agreement(self._cfg(), p)
        assert issues == []

    def test_rounds_mismatch_is_error(self, tmp_path):
        p = self._pyproject(tmp_path, **{
            "num-server-rounds": 2,          # 配置里是 100
            "local-epochs": 1,
            "learning-rate": 0.001,
            "batch-size": 64,
        })
        issues = check_app_config_agreement(self._cfg(), p)
        assert "C-APP" in codes(issues, LEVEL_ERROR)

    def test_float_tolerance(self, tmp_path):
        """0.001 在 TOML 里是浮点，不能因为表示误差判成不一致。"""
        p = self._pyproject(tmp_path, **{
            "num-server-rounds": 100,
            "local-epochs": 1,
            "learning-rate": 0.001,
            "batch-size": 64,
        })
        assert not has_errors(check_app_config_agreement(self._cfg(), p))

    def test_missing_key_is_warning(self, tmp_path):
        p = self._pyproject(tmp_path, **{"num-server-rounds": 100})
        issues = check_app_config_agreement(self._cfg(), p)
        assert "C-APP" in codes(issues, LEVEL_WARN)
        assert not has_errors(issues)

    def test_missing_file_is_warning(self, tmp_path):
        issues = check_app_config_agreement(
            self._cfg(), tmp_path / "nope.toml"
        )
        assert "C-APP" in codes(issues, LEVEL_WARN)

    def test_empty_app_config_is_warning(self, tmp_path):
        p = self._pyproject(tmp_path)
        issues = check_app_config_agreement(self._cfg(), p)
        assert "C-APP" in codes(issues, LEVEL_WARN)
