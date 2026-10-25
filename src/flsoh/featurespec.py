"""读特征表的口径版本（纯标准库）。

为什么单独一个模块
------------------
`feature_version` 决定一份特征表是「哪一版算法」算出来的。它要被三个地方读到：

1. `scripts/run_federated.py` / `scripts/run_local_baseline.py`——写进结果 JSON，
   让产物自己说清楚「我是拿哪一版特征训出来的」；
2. `scripts/rerun_all.py`——判 `--skip-existing` 该不该跳过。

训练脚本本来就依赖 numpy/torch，但 `rerun_all.py` 与它的测试跑在**纯标准库**
的 CI job 里，所以这段读取不能挂在 `flsoh.features`（那个模块 import numpy）。
单独放一个只依赖 `json`/`pathlib` 的小模块，是让两边都能用它的最省事的办法。

为什么这个字段值得单独存在
--------------------------
`--skip-existing` 原本只判「文件在不在」。`--out` 默认是 `results/`，而那里装的
可能是**上一版特征**的结果——于是重跑中断后带 `--skip-existing` 续跑会跳过全部
任务、打印「剩 0 个」、退出码 0，接着汇总出一张新旧混杂的表。这类错误不抛异常、
行数也对，只有数字是旧的。`feature_version` 是唯一能识破它的字段。
"""

from __future__ import annotations

import json
from pathlib import Path

#: 特征表旁挂的规格文件后缀。`features.parquet` → `features.spec.json`。
SPEC_SUFFIX = ".spec.json"


def spec_path_for(features: str | Path) -> Path:
    """`features.parquet` → 同目录的 `features.spec.json`。"""
    path = Path(features)
    return path.with_name(path.stem + SPEC_SUFFIX)


def read_feature_version(features: str | Path) -> str | None:
    """读规格文件里的 `feature_version`；读不出就返回 None。

    返回 None 表示「这份表没自报口径」——旧表就是这样。调用方**不能**把 None
    当成「一致」：拿一份不报口径的表去证明复现性，等于没查。
    """
    try:
        spec = json.loads(spec_path_for(features).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(spec, dict):
        return None
    value = spec.get("feature_version")
    return value if isinstance(value, str) else None
