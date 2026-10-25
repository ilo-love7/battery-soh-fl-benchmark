#!/usr/bin/env python3
"""充电会话 Schema v1（plan.md §8.1）。

App 侧与联邦仿真之间**唯一**的契约。现阶段没有真实 App 数据流，务实做法是
先定契约、再用模拟数据回填——所以这个模块的两件事必须分开：

1. **`validate_session`**：一份会话合不合法。这是入库前的闸门，不合法就拒收，
   不猜、不修补。猜错的代价见下面「为什么不宽容」。
2. **`session_fingerprint`**：一份会话的身份。App 重试上传会把同一份会话发两遍，
   入库时不按身份去重，「这个用户多跑了一个循环」就会被算成真实数据。

## 字段与边界

顶层必填：``user_id_hash`` / ``battery_id`` / ``session_id`` / ``start_ts`` /
``end_ts`` / ``chemistry_hint`` / ``sampling_hint`` / ``records``。

``records`` 里每条必填：``t`` / ``V`` / ``I`` / ``T`` / ``soc_est``。
``t`` 单位秒、``V`` 伏、``I`` 安（**负 = 放电**）、``T`` 摄氏、``soc_est`` 0–1。

数值边界不是「差不多就行」，每一条都对应一种会让特征静默变形的故障：

| 检查 | 放过会怎样 |
|---|---|
| ``t`` 严格递增 | 容量积分变成累加重复区间，Q(V) 整体偏高而曲线形状不变 |
| ``t`` 落在 ``[start_ts, end_ts]`` | 会话被拼进别的时间窗，按时间切 train/val 会切错 |
| ``V`` ∈ [1.5, 5.0] | 超出锂电物理区间，说明字段错位（把 I 写进了 V） |
| ``I`` ∈ [−20, 20] | 同上；20 A 是消费电子充电器的量级上限 |
| ``T`` ∈ [−30, 100] | 同上 |
| ``soc_est`` ∈ [0, 1] | 手机电量估计越界，通常是百分比忘了除 100 |
| ``end_ts − start_ts`` ∈ [1 s, 7 天] | 把「秒」写成了「毫秒」或反过来，段切分全废 |

## 为什么不宽容

- **未知字段只警告不报错。** 将来的 Schema v2 会加字段，现在报错等于宣告
  「这个仓库只认 v1」。
- **``schema_version`` 不匹配直接报错。** 反过来：v2 的语义可能改过（比如
  ``I`` 的符号约定翻转），按 v1 读会得到一份**结构合法、语义相反**的数据。
  这是最危险的一类错误——校验器全绿，特征全错。
- **``chemistry_hint`` 只在已知集合或 ``"unknown"`` 内。** 但它只是 hint：
  本仓库的 ΔQ(V) 特征不依赖化学体系，所以它不参与特征计算，只进元数据。
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

SCHEMA_VERSION = "v1"

SESSION_FIELDS: tuple[str, ...] = (
    "user_id_hash",
    "battery_id",
    "session_id",
    "start_ts",
    "end_ts",
    "chemistry_hint",
    "sampling_hint",
    "records",
)
RECORD_FIELDS: tuple[str, ...] = ("t", "V", "I", "T", "soc_est")

CHEMISTRY_HINTS = frozenset(
    {"LiCoO2", "LiFePO4", "NMC", "NCA", "LMO", "LTO", "unknown"}
)
SAMPLING_HINTS = frozenset({"variable", "fixed"})

# 数值边界。每条都在模块 docstring 的表里写了理由。
V_MIN, V_MAX = 1.5, 5.0
I_MIN, I_MAX = -20.0, 20.0
T_MIN, T_MAX = -30.0, 100.0
MIN_DURATION_S, MAX_DURATION_S = 1.0, 7 * 24 * 3600.0
MIN_HASH_HEX = 16
MAX_HASH_HEX = 64
MAX_ID_LEN = 128
MAX_RECORDS = 200_000


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(value: Any) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def _looks_like_hash(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if not MIN_HASH_HEX <= len(value) <= MAX_HASH_HEX:
        return False
    return all(c in "0123456789abcdef" for c in value)


def _is_id(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value) <= MAX_ID_LEN


def validate_session(session: Any, *, strict: bool = False) -> list[str]:
    """返回错误列表；空列表 = 合法。

    ``strict=True`` 时把「未知顶层字段」也升成错误。默认不升：Schema v2 会
    加字段，现在报错等于宣告这个仓库只认 v1。
    """
    errors: list[str] = []
    if not isinstance(session, dict):
        return [f"会话必须是 object，收到 {type(session).__name__}"]

    version = session.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        errors.append(
            f"schema_version={version!r} 不是 {SCHEMA_VERSION!r}；"
            "不同版本的字段语义可能改过（如电流符号约定），按旧版读会得到"
            "结构合法但语义相反的数据"
        )

    missing = [f for f in SESSION_FIELDS if f not in session]
    if missing:
        errors.append(f"缺必填字段：{missing}")
        # 缺字段时后续检查会连锁报错，先止损
        return errors

    unknown = sorted(set(session) - set(SESSION_FIELDS) - {"schema_version"})
    if unknown and strict:
        errors.append(f"未知顶层字段：{unknown}")

    if not _looks_like_hash(session["user_id_hash"]):
        errors.append(
            f"user_id_hash 必须是 {MIN_HASH_HEX}–{MAX_HASH_HEX} 位小写十六进制"
            f"（伪名化哈希），收到 {session['user_id_hash']!r:.60}"
        )
    for field in ("battery_id", "session_id"):
        if not _is_id(session[field]):
            errors.append(
                f"{field} 必须是 1–{MAX_ID_LEN} 字符的非空字符串，"
                f"收到 {session[field]!r:.60}"
            )

    start, end = session["start_ts"], session["end_ts"]
    ts_errors: list[str] = []
    for name, value in (("start_ts", start), ("end_ts", end)):
        if not _is_number(value) or not math.isfinite(float(value)):
            ts_errors.append(f"{name} 必须是有限数（unix 秒），收到 {value!r}")
    if not ts_errors:
        duration = float(end) - float(start)
        if not MIN_DURATION_S <= duration <= MAX_DURATION_S:
            ts_errors.append(
                f"end_ts − start_ts = {duration:.1f}s 不在 "
                f"[{MIN_DURATION_S:.0f}, {MAX_DURATION_S:.0f}]s；"
                "常见原因是秒/毫秒搞反，或把一整周拼成了一个会话"
            )
    errors.extend(ts_errors)
    # 时间戳本身不合法时不要再去比对 records 的时间窗，那会连锁报错
    ts_ok = not ts_errors

    if session["chemistry_hint"] not in CHEMISTRY_HINTS:
        errors.append(
            f"chemistry_hint={session['chemistry_hint']!r} 不在 "
            f"{sorted(CHEMISTRY_HINTS)}"
        )
    if session["sampling_hint"] not in SAMPLING_HINTS:
        errors.append(
            f"sampling_hint={session['sampling_hint']!r} 不在 {sorted(SAMPLING_HINTS)}"
        )

    errors.extend(_validate_records(session["records"], start, end, ts_ok))
    return errors


def _validate_records(records: Any, start: Any, end: Any,
                      ts_ok: bool = True) -> list[str]:
    if not isinstance(records, list):
        return [f"records 必须是数组，收到 {type(records).__name__}"]
    if not records:
        return ["records 为空：没有采样点的会话不携带任何信息"]
    if len(records) > MAX_RECORDS:
        return [f"records 有 {len(records)} 条，超过上限 {MAX_RECORDS}"]

    errors: list[str] = []
    prev_t: float | None = None
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            errors.append(f"records[{i}] 必须是 object")
            continue
        miss = [f for f in RECORD_FIELDS if f not in rec]
        if miss:
            errors.append(f"records[{i}] 缺字段 {miss}")
            continue
        t, v, cur, temp, soc = (rec[f] for f in RECORD_FIELDS)
        if not _finite(t):
            errors.append(f"records[{i}].t 不是有限数：{t!r}")
            continue
        t = float(t)
        if prev_t is not None and t <= prev_t:
            errors.append(
                f"records[{i}].t={t} 未严格大于上一条 {prev_t}；"
                "时间不递增时容量积分会累加重复区间，Q(V) 整体偏高而形状不变"
            )
            break
        prev_t = t
        for name, value, lo, hi in (
            ("V", v, V_MIN, V_MAX),
            ("I", cur, I_MIN, I_MAX),
            ("T", temp, T_MIN, T_MAX),
            ("soc_est", soc, 0.0, 1.0),
        ):
            if not _finite(value):
                errors.append(f"records[{i}].{name} 不是有限数：{value!r}")
            elif not lo <= float(value) <= hi:
                errors.append(
                    f"records[{i}].{name}={value} 不在 [{lo}, {hi}]"
                )

    if prev_t is not None and ts_ok:
        if not float(start) - 1e-6 <= prev_t <= float(end) + 1e-6:
            errors.append(
                f"records 的最后时间 {prev_t} 超出会话窗口 "
                f"[{start}, {end}]；按时间切 train/val 会切错"
            )
    return errors


def validate_sessions(sessions: Iterable[Any]) -> dict:
    """批量校验，返回计数 + 前若干条错误（不要把几千条错误全打印出来）。"""
    total = 0
    bad: list[dict] = []
    for i, s in enumerate(sessions):
        total += 1
        errs = validate_session(s)
        if errs:
            bad.append({"index": i, "errors": errs})
    return {
        "total": total,
        "valid": total - len(bad),
        "invalid": len(bad),
        # 只带前 20 条的摘要，完整错误由 check_schema.py --verbose 出
        "sample_errors": bad[:20],
    }


def canonical_json(obj: Any) -> str:
    """稳定序列化：键排序、无空白、浮点格式固定。

    指纹必须跨进程稳定——同一份会话在两台机器上算出同一个指纹，去重才有意义。
    ``json.dumps`` 的默认浮点表示依赖 repr，在极少数版本间会变，所以这里显式
    走 ``format(..., '.9g')``。
    """
    if isinstance(obj, dict):
        inner = ",".join(
            f"{json.dumps(k)}:{canonical_json(v)}" for k, v in sorted(obj.items())
        )
        return "{" + inner + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(canonical_json(v) for v in obj) + "]"
    if isinstance(obj, float):
        return format(obj, ".9g")
    if isinstance(obj, bool) or obj is None:
        return json.dumps(obj)
    if isinstance(obj, int):
        return str(obj)
    return json.dumps(obj)


def session_fingerprint(session: dict) -> str:
    """会话身份的 sha256（十六进制）。

    覆盖**全部内容**而不只是 ID：同一台设备重装 App 后 ``session_id`` 可能
    重复，而内容不同的概率可以忽略。反过来，只按 ID 去重会把「ID 相同、
    内容不同」的两份会话误判成重复——那种错误不会报错，只会让某个循环
    静默消失一个。
    """
    return hashlib.sha256(
        canonical_json(session).encode("utf-8")
    ).hexdigest()
