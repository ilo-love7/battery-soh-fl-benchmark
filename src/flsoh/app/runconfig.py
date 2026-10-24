"""把 flwr 的 run_config 取成一个普通 dict，并在 Python API 下兜底。

**为什么需要兜底**：flwr 1.41 的 `context.run_config` 只在使用 `flwr run`
（fab）时由 pyproject 的 `[tool.flwr.app.config]` 填充。直接调
`flwr.simulation.run_simulation(server_app, client_app, ...)` 时它是空的——
`run_simulation` 的签名里**没有任何参数能传 run_config**（1.41.0 实测：
只有 server_app / client_app / num_supernodes / backend_name /
backend_config / enable_tf_gpu_growth / verbose_logging）。

CI 的冒烟（`scripts/ci_smoke_flwr.py`）走的正是 Python API，于是
`context.run_config["num-server-rounds"]` 直接 KeyError。

兜底读的还是 `pyproject.toml` 的同一张表，所以**没有第二份超参**：
`flwr run` 时走 run_config，Python API 时走同一个文件的同一段。
`check_protocol.py` 的 `C-APP` 负责保证这张表与 `configs/mvp.toml` 一致。
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]

# 覆盖文件的路径。存在的理由：冒烟用 16 维合成数据，超参与真实特征表
# （1000 维）必然不同，不能靠改 pyproject 来做到——**测试不该改写被追踪的
# 文件**，一旦 finally 没执行（SIGKILL），仓库里就留下一个假配置。
ENV_RUN_CONFIG = "FLSOH_RUN_CONFIG"


def _read_config_table(path: Path) -> dict:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    table = data.get("tool", {}).get("flwr", {}).get("app", {}).get("config")
    if isinstance(table, dict):
        return dict(table)
    # 覆盖文件可以直接是扁平表（[tool.flwr.app.config] 那一层的内容）
    return {k: v for k, v in data.items() if not isinstance(v, dict)}


def _from_pyproject() -> dict:
    return _read_config_table(_ROOT / "pyproject.toml")


def app_run_config(context) -> dict:
    """返回本次运行的超参。

    优先级：
    1. flwr 给的 `context.run_config`（`flwr run` 路径）——**最高**
    2. `FLSOH_RUN_CONFIG` 指向的文件（只有冒烟脚本会用）
    3. `pyproject.toml` 的 `[tool.flwr.app.config]`（Python API 路径）

    为什么环境变量排第二而不是第一：`flwr run` 传进来的值是显式的、与这次
    运行绑定的；环境变量会留在 shell 里跨命令生效。一个忘了清掉的
    `FLSOH_RUN_CONFIG` 就能悄悄改掉一次正式实验的轮数，而日志里看不出来。

    1 与 3 读的是同一张表，所以不会出现「本地跑 100 轮、官方跑 2 轮」。
    """
    rc = getattr(context, "run_config", None)
    if rc:
        return dict(rc)
    override = os.environ.get(ENV_RUN_CONFIG)
    if override:
        cfg = _read_config_table(Path(override))
        if cfg:
            return cfg
    return _from_pyproject()
