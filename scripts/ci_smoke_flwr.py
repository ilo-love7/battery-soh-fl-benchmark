#!/usr/bin/env python3
"""冒烟测试：真跑一次 flwr 官方仿真运行时（M1 DOD ①）。

**为什么只能在 CI 跑**：flwr 1.41 的仿真后端只有 ray，而 ray 的 raylet
在 Windows + 中文主机名上启动即崩（见 scripts/run_federated.py 的 docstring）。
GitHub Actions 的 runner 主机名是 ASCII，所以那条路径在那边可用。

本脚本不验证精度，只验证**这条链路能通**：合成数据 → 划分产物 →
ClientApp/ServerApp 在 ray actor 里各自加载 → 聚合 → 全局评估。
精度验证在本地用 run_federated.py 做。

用法（CI 的 heavy-tests job 里）::

    python scripts/ci_smoke_flwr.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from flsoh.data import synthetic_feature_table  # noqa: E402
from flsoh.partition import partition_iid, split_test_cells  # noqa: E402

ROUNDS = 2
N_CLIENTS = 4


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="flsoh-smoke-"))

    table = synthetic_feature_table(
        n_cells=20, cycles_per_cell=10, input_dim=16, seed=0
    )
    feat_path = tmp / "features.parquet"
    table.frame.to_parquet(feat_path, index=False)

    cells = list(table.cells)
    test_cells, train_cells = split_test_cells(cells, n_test=4, seed=0)
    partition = partition_iid(train_cells, N_CLIENTS, seed=0)

    part_path = tmp / "partition.json"
    part_path.write_text(
        json.dumps(
            {
                "config": "smoke",
                "k": len(train_cells) // N_CLIENTS,
                "scheme": "iid",
                "alpha": None,
                "split_seed": 0,
                "test_cells": list(test_cells),
                "partition": partition,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # 子进程（ray actor）只能靠这两个变量拿到数据，见 client_app._ensure_loaded
    os.environ["FLSOH_FEATURES"] = str(feat_path)
    os.environ["FLSOH_PARTITION"] = str(part_path)

    from flwr.simulation import run_simulation

    from flsoh.app.client_app import app as client_app
    from flsoh.app.server_app import app as server_app

    print(f"合成数据：{len(table.cells)} 颗电池 / {len(table.frame)} 行")
    print(f"划分：{len(partition)} 个客户端，测试电池 {len(test_cells)} 颗")

    # run_simulation 从 app 目录的 pyproject.toml 读 run_config，没有参数能覆盖
    # num-server-rounds。冒烟不该跑 100 轮，所以临时改成 ROUNDS，跑完在 finally
    # 里改回——绝不留一个"配置说 100 轮、实际跑 2 轮"的状态。
    pyproject = ROOT / "pyproject.toml"
    original = pyproject.read_text(encoding="utf-8")
    patched = original.replace(
        "num-server-rounds = 100", f"num-server-rounds = {ROUNDS}"
    )
    if patched == original:
        print("FAIL 没找到 num-server-rounds = 100，配置格式变了？")
        return 2

    print(f"启动 ray 仿真，{ROUNDS} 轮", flush=True)
    pyproject.write_text(patched, encoding="utf-8")
    try:
        run_simulation(
            server_app, client_app, num_supernodes=len(partition), backend_name="ray"
        )
    finally:
        pyproject.write_text(original, encoding="utf-8")
        print("pyproject.toml 已恢复")

    print("SMOKE OK：flwr 仿真运行时跑通")
    return 0


if __name__ == "__main__":
    sys.exit(main())
