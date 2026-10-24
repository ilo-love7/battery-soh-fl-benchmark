"""`app_run_config` 的测试。

为什么值得单独测：这是**唯一**决定「跑多少轮、多少本地 epoch」的地方，
而它有两条路径（`flwr run` 的 run_config / Python API 的 pyproject 兜底）。
两条路径读到的值不一致时，任何一边都不会报错——只是跑的实验不一样。
纯标准库，默认 CI job 就能跑。
"""

from __future__ import annotations

import sys

import pytest

from flsoh.app.runconfig import ENV_RUN_CONFIG, app_run_config


class _Ctx:
    def __init__(self, run_config=None):
        self.run_config = run_config if run_config is not None else {}


def test_flwr_run_config_wins(tmp_path, monkeypatch):
    """`flwr run` 路径：context 里有值就必须用它的。"""
    override = tmp_path / "o.toml"
    override.write_text("num-server-rounds = 99\n", encoding="utf-8")
    monkeypatch.setenv(ENV_RUN_CONFIG, str(override))
    cfg = app_run_config(_Ctx({"num-server-rounds": 7, "input-dim": 3}))
    assert cfg["num-server-rounds"] == 7


def test_empty_run_config_falls_back_to_pyproject(monkeypatch):
    """Python API 路径：run_config 为空时读 pyproject 的同一张表。"""
    monkeypatch.delenv(ENV_RUN_CONFIG, raising=False)
    cfg = app_run_config(_Ctx({}))
    assert "num-server-rounds" in cfg
    assert "local-epochs" in cfg
    assert isinstance(cfg["num-server-rounds"], int)


def test_env_override_is_read(tmp_path, monkeypatch):
    override = tmp_path / "o.toml"
    override.write_text("num-server-rounds = 2\ninput-dim = 16\n", encoding="utf-8")
    monkeypatch.setenv(ENV_RUN_CONFIG, str(override))
    cfg = app_run_config(_Ctx({}))
    assert cfg["num-server-rounds"] == 2
    assert cfg["input-dim"] == 16


def test_env_override_accepts_nested_table(tmp_path, monkeypatch):
    """覆盖文件也可以直接写成 [tool.flwr.app.config] 那一层。"""
    override = tmp_path / "o.toml"
    override.write_text(
        "[tool.flwr.app.config]\nnum-server-rounds = 4\n", encoding="utf-8"
    )
    monkeypatch.setenv(ENV_RUN_CONFIG, str(override))
    assert app_run_config(_Ctx({}))["num-server-rounds"] == 4


def test_missing_override_file_falls_through(tmp_path, monkeypatch):
    """覆盖文件不存在时不能崩，要退回 pyproject。"""
    monkeypatch.setenv(ENV_RUN_CONFIG, str(tmp_path / "nope.toml"))
    cfg = app_run_config(_Ctx({}))
    assert "num-server-rounds" in cfg


def test_broken_override_file_falls_through(tmp_path, monkeypatch):
    override = tmp_path / "bad.toml"
    override.write_text("this is not toml = = =\n", encoding="utf-8")
    monkeypatch.setenv(ENV_RUN_CONFIG, str(override))
    cfg = app_run_config(_Ctx({}))
    assert "num-server-rounds" in cfg


def test_returns_a_copy_not_the_context_dict(monkeypatch):
    """不能把 flwr 的 run_config 对象直接交出去——调用方会往里写。"""
    monkeypatch.delenv(ENV_RUN_CONFIG, raising=False)
    ctx = _Ctx({"input-dim": 3})
    cfg = app_run_config(ctx)
    cfg["input-dim"] = 999
    assert ctx.run_config["input-dim"] == 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
