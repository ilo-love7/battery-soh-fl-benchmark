"""protocol 模块测试：每条校验规则都能报出来，且合法配置不误报。"""

import copy
from pathlib import Path

import pytest

from flsoh.protocol import (
    LEVEL_ERROR,
    LEVEL_WARN,
    ProtocolError,
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

    def test_warns_on_many_local_epochs(self):
        issues = validate_config(mutate("train.local_epochs", 5))
        assert "C-TRAIN" in codes(issues, LEVEL_WARN)
        assert not has_errors(issues)


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
