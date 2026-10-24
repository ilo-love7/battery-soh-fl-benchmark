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
INPUT_DIM = 16


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="flsoh-smoke-"))

    table = synthetic_feature_table(
        n_cells=20, cycles_per_cell=10, input_dim=INPUT_DIM, seed=0
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

    # `run_simulation` 的签名里没有 run_config（1.41.0 实测），`flwr run` 才
    # 会从 pyproject 填。所以走 `FLSOH_RUN_CONFIG` 覆盖文件（见
    # flsoh/app/runconfig.py）——**不去改 pyproject**：测试不该改写被追踪的
    # 文件，一旦 finally 没执行，仓库里就留下一个假配置（这件事真发生过）。
    #
    # input-dim 必须跟着合成数据走：真实特征表是 1000 维，合成的是 16 维。
    override = tmp / "smoke-runconfig.toml"
    override.write_text(
        "\n".join(
            [
                f"num-server-rounds = {ROUNDS}",
                "local-epochs = 2",
                f"input-dim = {INPUT_DIM}",
                'optimizer = "adam"',
                "learning-rate = 0.001",
                "proximal-mu = 0.0",
                "batch-size = 8",
                "seed = 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    os.environ["FLSOH_RUN_CONFIG"] = str(override)

    print(f"启动 ray 仿真，{ROUNDS} 轮", flush=True)
    run_simulation(
        server_app, client_app, num_supernodes=len(partition), backend_name="ray"
    )

    print("SMOKE OK：flwr 仿真运行时跑通")
    return 0


if __name__ == "__main__":
    sys.exit(main())
