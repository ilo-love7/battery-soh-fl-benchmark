#!/usr/bin/env python3
"""本地特征提取：原始采样记录 → ΔQ(V) 特征向量（plan.md §8.2）。

## 这个模块存在的理由

plan.md §8.2 要求「与联邦仿真**共用同一份代码**」。这句话的可执行含义是：
App 侧和实验室侧从 (V, Q) 曲线往后的每一步必须是**同一个函数**。所以本模块
只做「记录 → (V, Q) 曲线」这一件事，之后立刻交给
``flsoh.features.delta_q_curve`` / ``delta_q_features``——正是
``build_features.py`` 调用的那两个。

边界划在这里而不是各写一份，是因为两类数据的**失效方式不同**：

| | 实验室路径（MATR） | App 路径（本模块） |
|---|---|---|
| 采样 | 固定步长、已对齐 | 可变步长、有丢点 |
| 电流 | 已知符号 | 要自己判断充/放 |
| 容量 | 文件里直接给 ``QDischarge`` | 必须对 I 积分 |
| 电压 | 单调 | 静置时松弛，会回跳 |

后三行都是 App 路径独有的错误来源，而它们的共同特征是**不抛异常**：切错段、
积分错符号、静置点混进曲线，得到的特征向量长度和形状都正常，只是数值整体偏移。

## 三个决定

1. **放电判据是电流符号，且要过一个静止门限。** ``I < 0`` 判放电（Schema v1
   的约定）。但「几乎为 0」的采样点既不是充电也不是放电，把它们算进放电段会
   让容量积分掺进静置时间——而静置期间电压在松弛，曲线会出现垂直跳变。
   ``REST_THRESHOLD_A`` 就是干这个的。
2. **容量从每个放电段的起点重新积分。** 对齐 ``QDischarge`` 的语义（每循环
   的放电容量），不是「从上车开始的总放电量」。
3. **(V, Q) 只取段内放电点。** 见上：段内已经没有静置点了，这一条是防御性的，
   但它保证 ``V`` 在段内基本单调——``delta_q_curve`` 对重复电压的行为依赖
   ``np.interp`` 的取值顺序，不值得依赖。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flsoh.features import DataContractError, delta_q_curve, delta_q_features

#: Schema v1 的电流符号约定：负 = 放电。
DISCHARGE_SIGN = -1.0
#: |I| 低于这个值视为静置，断开当前段。0.05 A 对消费电子充电器是噪声量级。
REST_THRESHOLD_A = 0.05


@dataclass(frozen=True)
class Segment:
    """一段连续的放电采样。"""

    start: int          # 在 records 里的起始下标（含）
    stop: int           # 结束下标（不含）
    t0: float           # 段首时间（unix 秒）
    t1: float           # 段尾时间
    mean_current: float # 段内平均电流（A，负值）

    @property
    def n_points(self) -> int:
        return self.stop - self.start

    @property
    def duration_s(self) -> float:
        return self.t1 - self.t0


def _records_to_arrays(records: list[dict]) -> tuple[np.ndarray, ...]:
    t = np.array([float(r["t"]) for r in records], dtype=float)
    v = np.array([float(r["V"]) for r in records], dtype=float)
    cur = np.array([float(r["I"]) for r in records], dtype=float)
    temp = np.array([float(r["T"]) for r in records], dtype=float)
    return t, v, cur, temp


def split_segments(
    records: list[dict],
    *,
    current_sign: float = DISCHARGE_SIGN,
    rest_threshold_a: float = REST_THRESHOLD_A,
    min_points: int = 8,
    min_duration_s: float = 60.0,
) -> list[Segment]:
    """把一段会话的采样按电流符号切成连续段。

    **为什么段内不允许静置**：静置期间电压松弛而容量不变，(V, Q) 上会出现
    一段垂线；``delta_q_curve`` 用 ``np.interp`` 处理重复电压，取哪个点取决于
    排序的稳定性——不值得把特征正确性押在这上面。

    短段直接丢掉并**不静默**：调用方拿不到被丢了多少，所以
    :func:`split_segments_report` 会一并给出统计。
    """
    if not records:
        raise DataContractError("records 为空")
    t, _v, cur, _temp = _records_to_arrays(records)
    active = cur * current_sign > rest_threshold_a

    segments: list[Segment] = []
    i = 0
    n = len(records)
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and active[j + 1]:
            j += 1
        stop = j + 1
        seg = Segment(
            start=i,
            stop=stop,
            t0=float(t[i]),
            t1=float(t[j]),
            mean_current=float(np.mean(cur[i:stop])),
        )
        if seg.n_points >= min_points and seg.duration_s >= min_duration_s:
            segments.append(seg)
        i = stop
    return segments


def split_segments_report(
    records: list[dict], **kwargs
) -> tuple[list[Segment], dict]:
    """:func:`split_segments` 加上「丢了多少」的统计。

    丢段必须可见：一个采样率太低的会话可能整段都不到 ``min_points``，
    静默丢掉会让「这个用户贡献了 0 个循环」和「这个用户没上传」看起来一样。
    """
    t, _v, cur, _temp = _records_to_arrays(records)
    active = cur * kwargs.get("current_sign", DISCHARGE_SIGN) > kwargs.get(
        "rest_threshold_a", REST_THRESHOLD_A
    )
    # 先数一遍全部候选段（不分长短），再按门限过滤
    candidates: list[tuple[int, int]] = []
    i = 0
    n = len(records)
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and active[j + 1]:
            j += 1
        candidates.append((i, j + 1))
        i = j + 1

    kept = split_segments(records, **kwargs)
    kept_starts = {s.start for s in kept}
    dropped = [
        {"start": a, "stop": b, "points": b - a, "duration_s": float(t[b - 1] - t[a])}
        for a, b in candidates
        if a not in kept_starts
    ]
    return kept, {
        "n_records": n,
        "n_candidates": len(candidates),
        "n_kept": len(kept),
        "dropped": dropped,
    }


def integrate_capacity(
    t: np.ndarray, current: np.ndarray, *, current_sign: float = DISCHARGE_SIGN
) -> np.ndarray:
    """梯形积分累计容量（Ah），从 0 开始。

    ``t`` 单位秒、``current`` 安。只累计与 ``current_sign`` 同号的部分——
    段内理论上全是同号点，这一条是防御性的（混进一个静置点不该让容量倒退）。

    **为什么是梯形而不是左端点**：可变采样率下相邻两点间隔差很多，左端点
    等效于把每个采样当成阶跃持续到下一个，采样稀的那一段会被高估。
    """
    t = np.asarray(t, dtype=float)
    current = np.asarray(current, dtype=float)
    if t.shape != current.shape:
        raise DataContractError(
            f"t 与 current 长度不一致：{t.shape} vs {current.shape}"
        )
    if t.size < 2:
        raise DataContractError("至少需要 2 个采样点才能积分")
    if np.any(np.diff(t) <= 0):
        raise DataContractError("t 必须严格递增（Schema v1 已校验，这里是兜底）")
    signed = current * current_sign
    signed = np.where(signed > 0, signed, 0.0)
    dt = np.diff(t)
    q = np.concatenate([[0.0], np.cumsum(0.5 * (signed[1:] + signed[:-1]) * dt)])
    return q / 3600.0  # A·s → Ah


def segment_pair(records: list[dict], segment: Segment) -> tuple[np.ndarray, np.ndarray]:
    """一个放电段 → 原始 ``(voltage, capacity)`` 数组（**未**插值）。

    返回的是实验室路径同一种东西：``build_features.py`` 从 MATR 拿到的
    ``(V, Q)`` 就是这个形式。往后交给
    ``flsoh.features.delta_q_curve`` / ``delta_q_features``——与实验室路径
    完全同一批函数，这就是 plan §8.2「共用同一份代码」的可执行含义。
    """
    t, v, cur, _temp = _records_to_arrays(records)
    sl = slice(segment.start, segment.stop)
    q = integrate_capacity(t[sl], cur[sl])
    return v[sl], q


def segment_curve(
    records: list[dict], segment: Segment, v_grid: np.ndarray
) -> np.ndarray:
    """一个放电段 → Q(V)，内部就是 ``flsoh.features.delta_q_curve``。"""
    v, q = segment_pair(records, segment)
    return delta_q_curve(v, q, v_grid)


def session_curves(session: dict, v_grid: np.ndarray) -> list[np.ndarray]:
    """一份会话里所有放电段的 Q(V) 曲线（未差分）。"""
    records = session["records"]
    return [segment_curve(records, seg, v_grid)
            for seg in split_segments(records)]


def _spans_grid(voltage: np.ndarray, v_grid: np.ndarray) -> bool:
    """电压曲线是否覆盖整个网格。

    这是 ``delta_q_curve`` 的前置条件，在这里提前判而不是等它抛：
    ``delta_q_features`` 对整个列表一次性插值，抛异常时不说清是哪一段，
    而「哪个用户的哪个循环不见了」必须逐条可读。容差 1e-9 与
    ``delta_q_curve`` 保持一致。
    """
    lo, hi = float(voltage.min()), float(voltage.max())
    return not (v_grid.min() < lo - 1e-9 or v_grid.max() > hi + 1e-9)


def battery_delta_q(
    sessions: list[dict],
    v_grid: np.ndarray,
    *,
    reference_index: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """一颗电池的多次会话 → ΔQ(V) 特征矩阵 ``[n_cycles, len(v_grid)]``。

    ``sessions`` 必须属于同一颗电池且**按时间排序**——参考循环取最早那段，
    与 ``build_features.py`` 的 ``--ref-cycle`` 语义对齐。返回第二个元素是
    被跳过的会话 ID 及原因（采样不足、曲线覆盖不住电压窗口）。

    覆盖不住窗口的会话**逐条记下来**，不静默跳过：静默跳过会让「这个用户的
    第 3 个循环不见了」在特征表里完全看不出来。
    """
    if not sessions:
        raise DataContractError("sessions 为空")
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    skipped: list[str] = []
    for s in sessions:
        sid = str(s.get("session_id", "?"))
        try:
            segs = split_segments(s["records"])
        except DataContractError as exc:
            skipped.append(f"{sid}: {exc}")
            continue
        if not segs:
            skipped.append(f"{sid}: 没有够长的放电段")
            continue
        # 一段会话取最长那个放电段：手机可能在一次插电里放多次电，而
        # build_features.py 的「一个循环一条曲线」对应的是完整放电。
        longest = max(segs, key=lambda x: x.duration_s)
        try:
            v, q = segment_pair(s["records"], longest)
        except DataContractError as exc:
            skipped.append(f"{sid}: {exc}")
            continue
        if not _spans_grid(v, v_grid):
            skipped.append(
                f"{sid}: 段内电压 [{v.min():.3f}, {v.max():.3f}] 覆盖不住网格 "
                f"[{v_grid.min():.3f}, {v_grid.max():.3f}]"
            )
            continue
        pairs.append((v, q))
    if not pairs:
        raise DataContractError(
            f"一条曲线都没抽出来：{len(skipped)} 个会话全部跳过"
        )
    if reference_index >= len(pairs):
        raise DataContractError(
            f"reference_index={reference_index} 越界（只抽出 {len(pairs)} 条曲线）"
        )
    # 与 build_features.py 逐字对应：cycles 是 (V, Q) 对的列表
    return delta_q_features(pairs, v_grid, reference_index=reference_index), skipped


# ---------------------------------------------------------------------------
# App 路径的保真度：这条链比实验室路径多付多少
# ---------------------------------------------------------------------------


def degrade_to_records(
    voltage: np.ndarray,
    capacity: np.ndarray,
    *,
    t_start: float = 0.0,
    current_a: float = -1.2,
    sample_dt_s: float = 30.0,
    voltage_noise_v: float = 0.0,
    current_noise_a: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """把一条实验室 (V, Q) 曲线降级成 Schema v1 的采样记录。

    用来量化「App 路径比实验室路径多付多少」：喂进来的曲线已经是干净形式，
    出去的是带可变采样与噪声的记录，再喂回 :func:`segment_curve` 就能测出
    保真度。降级项逐项可关，便于区分「采样率」与「测量噪声」各自的代价。

    ``sample_dt_s`` 是**名义**间隔：实际间隔乘一个 [0.5, 1.5] 的随机因子，
    模拟手机的可变采样率（Schema v1 的 ``sampling_hint="variable"``）。
    """
    voltage = np.asarray(voltage, dtype=float)
    capacity = np.asarray(capacity, dtype=float)
    if voltage.shape != capacity.shape:
        raise DataContractError("voltage 与 capacity 长度不一致")
    if voltage.size < 2:
        raise DataContractError("至少需要 2 个点")
    rng = rng or np.random.default_rng(0)

    # 先丢掉「容量没变」的相邻点。它们对 Q(V) 没有信息量，却会让下面反推出的
    # 时间步长为 0——而 t 重复会让 integrate_capacity 抛错（「t 必须严格递增」），
    # 也就是这份函数自己产出的记录违反了它自己的契约。
    # 真实放电曲线的末端常有一小段容量平台（静置或采样重复），所以这不是
    # 假想输入：b1c0 循环 5 的 445 个点里就有重复容量。
    unchanged = np.concatenate([[False], np.diff(capacity) == 0])
    if unchanged.any():
        voltage = voltage[~unchanged]
        capacity = capacity[~unchanged]
    if voltage.size < 2:
        raise DataContractError("去掉容量平台后不足 2 个点")

    # 容量 → 时间：按恒定电流反推 dQ/dt，再积分。这样降级后的记录与原始
    # (V, Q) 描述的是同一次放电，而不是两条不同的放电。
    dq = np.diff(capacity) * 3600.0                     # Ah → A·s
    dt = np.abs(dq / current_a)
    t = np.concatenate([[0.0], np.cumsum(dt)])

    keep_t: list[float] = [t[0]]
    keep_v: list[float] = [voltage[0]]
    keep_i: list[float] = [current_a]
    next_at = 0.0
    for k in range(1, t.size):
        if t[k] >= next_at:
            keep_t.append(float(t[k]))
            keep_v.append(float(voltage[k]))
            keep_i.append(float(current_a))
            next_at = t[k] + sample_dt_s * float(rng.uniform(0.5, 1.5))
    if len(keep_t) < 2:
        # 采样太稀，只剩一个点：加密到至少两点，否则段切分直接丢
        keep_t.append(float(t[-1]))
        keep_v.append(float(voltage[-1]))
        keep_i.append(float(current_a))

    t_arr = np.array(keep_t) + t_start
    v_arr = np.array(keep_v)
    if voltage_noise_v:
        v_arr = v_arr + rng.normal(0.0, voltage_noise_v, size=v_arr.shape)
    i_arr = np.array(keep_i)
    if current_noise_a:
        i_arr = i_arr + rng.normal(0.0, current_noise_a, size=i_arr.shape)
    soc = np.linspace(1.0, 0.0, t_arr.size)
    return [
        {"t": float(t_arr[k]), "V": float(v_arr[k]), "I": float(i_arr[k]),
         "T": 25.0, "soc_est": float(soc[k])}
        for k in range(t_arr.size)
    ]


def app_roundtrip_error(
    voltage: np.ndarray,
    capacity: np.ndarray,
    v_grid: np.ndarray,
    *,
    sample_dt_s: float = 30.0,
    voltage_noise_v: float = 0.0,
    current_noise_a: float = 0.0,
    seed: int = 0,
) -> dict:
    """实验室曲线走一遍 App 路径后，Q(V) 差多少。

    返回 RMSE / 最大偏差 / 相对偏差（相对曲线自身的动态范围）。这两个数
    是 report.md「模拟 vs 部署差距」一节的直接输入——没有它们，那一节只能
    写「App 数据更脏」，而那是无法验证的陈述。

    ``n_segments`` 是**产出了可用曲线**的段数，所以它与 ``relative == inf``
    等价：取不到曲线时报 0，不报「切出来了但没用」的那个 1。
    """
    rng = np.random.default_rng(seed)
    clean = delta_q_curve(np.asarray(voltage, float), np.asarray(capacity, float),
                          v_grid)
    records = degrade_to_records(
        voltage, capacity, current_a=-1.2, sample_dt_s=sample_dt_s,
        voltage_noise_v=voltage_noise_v, current_noise_a=current_noise_a,
        rng=rng,
    )
    segs = split_segments(records, min_points=2, min_duration_s=0.0)
    if not segs:
        return {"rmse": float("inf"), "max_abs": float("inf"),
                "relative": float("inf"), "n_records": len(records),
                "n_segments": 0}
    longest = max(segs, key=lambda s: s.duration_s)
    try:
        recovered = segment_curve(records, longest, v_grid)
    except DataContractError:
        # 采样稀到段内电压覆盖不住网格。**必须报出来而不是给一个数**：
        # 这种情况下「App 路径的损失」不是大一点，是这一循环整个不存在。
        # n_segments 记 0 而不是 1——段确实被切出来了，但它产不出一条可用
        # 曲线，报 1 会让读的人以为恢复出了一个循环。
        return {"rmse": float("inf"), "max_abs": float("inf"),
                "relative": float("inf"), "n_records": len(records),
                "n_segments": 0}
    diff = recovered - clean
    span = float(clean.max() - clean.min())
    return {
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "max_abs": float(np.max(np.abs(diff))),
        "relative": float(np.max(np.abs(diff)) / span) if span > 0 else float("inf"),
        "n_records": len(records),
        "n_segments": len(segs),
    }
