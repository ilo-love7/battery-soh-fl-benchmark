#!/usr/bin/env python3
"""会话入库：校验 → 去重 → 组装特征表。

## 为什么入库要单独一层

App 侧的数据不是「一个 parquet 文件」，而是一条条上传的会话：同一份会被重试
上传好几遍，中途会有半条坏记录，还有版本不一致的旧客户端。这一层要做的事只有
三件，而每一件**做错的代价都不体现在报错上**：

1. **校验**：不合法的会话拒收，不猜、不修补（规则在 :mod:`flsoh.schema`）。
   猜错的代价是「结构合法、语义相反」的数据进了训练集。
2. **去重**：按 :func:`flsoh.schema.session_fingerprint` 去重。App 重试上传
   很常见；不去重的话，「这个用户多跑了一个循环」会被当成真实数据，
   而客户端样本量因此虚高——联邦加权平均的权重直接算错。
3. **组装**：把会话变成 ``(cell_id, cycle, soh, f0..fN)`` 的行，喂给训练。
   ΔQ(V) 走的是 :mod:`flsoh.segments` 的 App 路径——**与实验室路径同一份代码**
   （plan §8.2），不是另写一份「差不多的」。

## App 路径没有「循环号」

Schema v1 里没有 cycle 字段，只有时间戳。所以这里用**时间序序号**填 ``cycle``
列：它只用于按时间切 train/val（:func:`flsoh.data.split_cycles_by_time`），
不参与任何特征计算。这一点必须写明——如果哪天有人拿这一列去对实验室路径的
循环号，两边对不上而他不知道原因。

## 标签从哪来

模拟数据有真值，真实 App 数据没有。所以 ``sessions_to_frame`` 要求显式传入
``truth``（``session_id → soh``），缺一条就报错而不是填 NaN：**没有标签的行
进训练集会静默地把损失拉向 0**，比少几行危险得多。真实部署的标签来源（首会话
基准 + 库仑计容量）不在本模块范围内。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from flsoh.schema import (
    SCHEMA_VERSION,
    session_fingerprint,
    validate_session,
)

#: 错误摘要最多带几条。几千条错误全打印出来没人看，反而会把真正的原因埋掉。
MAX_ERRORS = 20


class IngestError(Exception):
    """入库阶段的致命错误（文件读不了、JSON 坏了、标签缺了）。"""


def read_sessions_jsonl(path: str | Path) -> list[dict]:
    """读一行一个会话的 JSONL。

    **坏行直接抛错并带行号**，不跳过。理由：跳过的坏行会让「入库了多少条」
    这个数字悄悄变小，而下游看到的是一个完全正常的表。要容忍坏行，调用方
    应该先自己按行切再逐条传给 :func:`ingest_sessions`。
    """
    path = Path(path)
    if not path.is_file():
        raise IngestError(f"会话文件不存在：{path}")
    sessions: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sessions.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise IngestError(f"{path}:{lineno} 不是合法 JSON：{exc}") from exc
    return sessions


def ingest_sessions(sessions: Iterable[Any], *, strict: bool = False) -> dict:
    """校验 + 去重，返回计数与被接收的会话。

    返回字段：

    - ``n_total`` / ``n_accepted`` / ``n_invalid`` / ``n_duplicate``
    - ``accepted``：去重后的会话（保持输入顺序）
    - ``errors``：前 :data:`MAX_ERRORS` 条不合法会话的摘要

    ``strict=True`` 透传给 :func:`validate_session`，把「未知字段」也当错误。
    默认不严：将来的 Schema v2 会加字段。
    """
    accepted: list[dict] = []
    errors: list[dict] = []
    seen: dict[str, str] = {}
    n_total = n_invalid = n_duplicate = 0

    for session in sessions:
        n_total += 1
        problems = validate_session(session, strict=strict)
        if problems:
            n_invalid += 1
            if len(errors) < MAX_ERRORS:
                errors.append(
                    {
                        "session_id": _safe_id(session),
                        "errors": problems,
                    }
                )
            continue
        fingerprint = session_fingerprint(session)
        if fingerprint in seen:
            # 指纹相同 = 内容逐字相同。记下第一个的来源，方便查是谁在重试。
            n_duplicate += 1
            if len(errors) < MAX_ERRORS:
                errors.append(
                    {
                        "session_id": _safe_id(session),
                        "duplicate_of": seen[fingerprint],
                        "errors": [],
                    }
                )
            continue
        seen[fingerprint] = _safe_id(session)
        accepted.append(session)

    return {
        "schema_version": SCHEMA_VERSION,
        "n_total": n_total,
        "n_accepted": len(accepted),
        "n_invalid": n_invalid,
        "n_duplicate": n_duplicate,
        "accepted": accepted,
        "errors": errors,
    }


def _safe_id(session: Any) -> str:
    """从一份可能残缺的会话里尽量取出 ID，取不到就标 ``<unknown>``。"""
    if isinstance(session, dict):
        value = session.get("session_id")
        if isinstance(value, str):
            return value[:128]
    return "<unknown>"


def sessions_to_frame(
    sessions: list[dict],
    *,
    v_grid,
    truth: dict[str, float],
):
    """会话 → 特征表 DataFrame（``cell_id`` / ``cycle`` / ``soh`` / ``f0..fN``）。

    ``truth`` 是 ``session_id → soh``。缺任何一条被用到的会话就抛
    :class:`IngestError`——见模块 docstring「标签从哪来」。

    参考循环取**每颗电池时间最早的那份会话**，特征 = ``Qd_k(V) − Qd_ref(V)``，
    参考会话自身不产生行（``ΔQ ≡ 0``，没有信息量）。
    """
    import numpy as np
    import pandas as pd

    from flsoh.features import CELL_COL, CYCLE_COL, SOH_COL
    from flsoh.segments import battery_delta_q

    if not sessions:
        raise IngestError("没有可用的会话")

    by_battery: dict[str, list[dict]] = {}
    for session in sessions:
        by_battery.setdefault(session["battery_id"], []).append(session)
    for battery in by_battery.values():
        battery.sort(key=lambda s: (s["start_ts"], s["session_id"]))

    missing = [
        s["session_id"]
        for group in by_battery.values()
        for s in group
        if s["session_id"] not in truth
    ]
    if missing:
        raise IngestError(
            f"{len(missing)} 份会话没有真值标签（前 5 条：{missing[:5]}）。"
            "没有标签的行进训练集会静默把损失拉向 0，所以这里直接拒绝。"
        )

    rows: list[dict] = []
    dropped: list[str] = []
    for battery_id, group in sorted(by_battery.items()):
        curves, skipped = battery_delta_q(group, v_grid)
        dropped.extend(f"{battery_id}: {msg}" for msg in skipped)
        # 跳过的会话不参与排序，所以用「没被跳过」的下标去取标签。
        used = [s for s in group[1:] if not _was_skipped(s, skipped)]
        if curves.shape[0] != len(used):
            raise IngestError(
                f"{battery_id}: 特征行数 {curves.shape[0]} 与可用会话数 "
                f"{len(used)} 不一致（跳过 {len(skipped)} 份）。"
                "行与标签对不上时，任何指标都没有意义。"
            )
        for offset, session in enumerate(used):
            row = {
                CELL_COL: battery_id,
                # 时间序序号：只用于按时间切 train/val，不参与特征。
                CYCLE_COL: offset + 1,
                SOH_COL: float(truth[session["session_id"]]),
            }
            for i, value in enumerate(curves[offset]):
                row[f"f{i}"] = float(value)
            rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise IngestError("一份会话都没能算出特征；先看 dropped 里的原因")
    return frame, dropped


def _was_skipped(session: dict, skipped: list[str]) -> bool:
    """``battery_delta_q`` 的跳过消息以 ``<session_id>: <原因>`` 开头。"""
    return any(
        msg.split(":", 1)[0] == session["session_id"] for msg in skipped
    )


def truth_from_json(payload: dict | list) -> dict[str, float]:
    """把 ``simulate_sessions.py --truth`` 的产物转成 ``session_id → soh``。"""
    items = payload["sessions"] if isinstance(payload, dict) else payload
    out: dict[str, float] = {}
    for item in items:
        session_id = item.get("session_id")
        soh = item.get("soh")
        if not isinstance(session_id, str) or soh is None:
            continue
        out[session_id] = float(soh)
    return out


def frame_to_numpy(frame) -> tuple:
    """``(特征矩阵, cell_id 数组, cycle 数组, soh 数组)``，给不方便用 DataFrame 的地方。"""
    import numpy as np

    feats = [c for c in frame.columns if c.startswith("f") and c[1:].isdigit()]
    matrix = np.asarray(frame[feats].to_numpy(dtype=float))
    return (
        matrix,
        frame["cell_id"].to_numpy(),
        frame["cycle"].to_numpy(),
        frame["soh"].to_numpy(dtype=float),
    )
