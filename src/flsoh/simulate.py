#!/usr/bin/env python3
"""模拟 App 会话生成器：在 Schema v1 契约下造出**带真值**的可训练数据。

## 为什么需要它

真实的 App 数据不存在，而端到端联调（``scripts/run_e2e.py``）不能等。模拟
数据的用途**不是**「证明方法有效」——它只能验证**管道**是否接对：
原始记录 → 分段 → ΔQ(V) → 训练 → 聚合。任何一环接错，模拟数据上的 SOH
就恢复不出来。

所以生成器必须同时输出**真值**。没有真值，「恢复出来了」这句话无法验证，
只能看着指标像模像样地说「效果不错」——这正是本项目最想避免的那类陈述。

## 物理模型（刻意简单）

- 开路电压 ``OCV(soc) = v_lo + (v_hi − v_lo)·soc^γ``，γ≈0.35（低 SOC 端更陡，
  与 LFP 的放电曲线定性一致）
- 端电压 ``V = OCV(soc) + I·R``，``I < 0`` 为放电（与 MATR 同号），所以放电时
  端电压低于开路电压
- 容量随循环衰减 ``Q(c) = Q0·(1 − fade·√(c/C))``，每颗电池的 ``fade`` 不同
- 内阻随容量衰减增大 ``R = R0·(1 + 0.6·(1 − Q/Q0))``——这一项让 ΔQ(V) 不只是
  「整条曲线乘一个系数」，否则模型只需要学一个标量，联调也就测不出什么
- 采样间隔有抖动（``sampling_hint = "variable"``），电压/电流/温度都有噪声
- 每段放电前后各插一小段静置（``I = 0``），用来验证分段器**确实按电流断开**，
  而不是把整条记录当成一段

**故意不模拟**：温度对容量的影响、倍率效应、充放电迟滞、内阻随 SOC 变化、
容量再生（calendar/cycle 混合老化）。理由：这些都会让真值本身变得不可知，
而本模块唯一的产出就是可知的真值。要研究它们，需要真实数据，不需要更花哨的
模拟器。

## 真值是什么

``truth`` 里每条对应一份会话：``capacity_ah`` 是该循环的实际可放容量，
``soh = capacity_ah / capacity_ah(首个循环)``。**相对首个循环**而不是相对出厂
容量——实验室路径的定义是 ``SOH_k = QD_k / QD_ref``（ref = 首个循环），两条
路径的标签必须同尺度，否则联调出来的指标无法与 M1–M4 的结果放在一起看。

（``CellSpec.q0_ah`` 是出厂容量，只用来算 ``capacity_ah``；因为模拟从第 1 个
循环就开始衰减，首循环容量本身就低于 ``q0_ah``。）
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: 开路电压的上下界。**故意比 1000 点网格 [2.0, 3.5] 略宽**：
#: 端电压 = OCV + I·R 在放电时会被拉低，若 OCV 下界正好取 2.0，端电压就
#: 落到 1.98 而网格下界 2.0 覆盖不到 → 每一行都被判「覆盖不住」。
#:
#: 上界 3.62 V 比 LFP 的静置电压高一点，是给内阻留余量：重度老化时
#: ``R`` 会涨到 0.064 Ω，1C 放电的压降是 0.070 V，若 OCV 上界只有 3.56，
#: 端电压最高只能到 3.49 V——**差 10 mV 就覆盖不住网格上界**，而那几行会
#: 被整行丢弃。用「曲线略微不真实」换「不丢数据」，在模拟器里是划算的。
OCV_LO_V = 2.02
OCV_HI_V = 3.62

#: 低 SOC 端的陡峭程度。0.35 让曲线在 2.0–2.4 V 区间密集、3.2–3.4 V 区间平坦，
#: 与 LFP 放电曲线的定性形状一致。
DEFAULT_GAMMA = 0.35

#: 静置段的采样点数（前后各这么多）。不用 1 个点：单点静置在「连续同号才算一段」
#: 的分段逻辑里和噪声无法区分。
REST_POINTS = 4

#: 缺省噪声水平。电压噪声 2 mV 是消费电子 BMS 的典型量级。
DEFAULT_NOISE_V = 2e-3
DEFAULT_NOISE_I = 0.01
DEFAULT_NOISE_T = 0.3


@dataclass(frozen=True)
class CellSpec:
    """一颗模拟电池的物理参数。

    ``fade`` 是循环寿命终点时的容量损失比例（``√(c/C)`` 在 ``c=C`` 处取 1），
    所以 ``fade=0.12`` 表示跑完全部循环后容量剩 88%。
    """

    battery_id: str
    q0_ah: float = 1.10
    fade: float = 0.12
    r0_ohm: float = 0.030
    gamma: float = DEFAULT_GAMMA
    discharge_a: float = 1.10

    def __post_init__(self) -> None:
        if self.q0_ah <= 0:
            raise ValueError(f"q0_ah 必须为正，收到 {self.q0_ah}")
        if not 0.0 <= self.fade < 1.0:
            raise ValueError(f"fade 必须在 [0, 1) 内，收到 {self.fade}")
        if self.r0_ohm < 0:
            raise ValueError(f"r0_ohm 不能为负，收到 {self.r0_ohm}")
        if self.discharge_a <= 0:
            raise ValueError(f"discharge_a 必须为正（符号由模块内部取），收到 {self.discharge_a}")


def open_circuit_voltage(soc, *, gamma: float = DEFAULT_GAMMA) -> np.ndarray:
    """``OCV(soc) = OCV_LO + (OCV_HI − OCV_LO)·soc^γ``，soc 截到 [0, 1]。"""
    soc = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
    return OCV_LO_V + (OCV_HI_V - OCV_LO_V) * np.power(soc, gamma)


def capacity_at(spec: CellSpec, cycle_no: int, total_cycles: int) -> float:
    """第 ``cycle_no`` 个循环的可放容量（Ah）。``cycle_no`` 从 1 开始。

    用平方根衰减而不是线性：线性衰减会让「每循环掉多少」在整条轨迹上是常数，
    模型只要学会平均斜率就够，掩盖了「衰减先快后慢」这个真实特征。
    """
    if total_cycles <= 0:
        raise ValueError(f"total_cycles 必须为正，收到 {total_cycles}")
    if cycle_no < 1:
        raise ValueError(f"cycle_no 从 1 开始，收到 {cycle_no}")
    progress = min(float(cycle_no) / float(total_cycles), 1.0)
    return spec.q0_ah * (1.0 - spec.fade * float(np.sqrt(progress)))


def internal_resistance(spec: CellSpec, capacity_ah: float) -> float:
    """内阻随容量衰减增大。``capacity_ah == q0_ah`` 时等于 ``r0_ohm``。"""
    loss = 1.0 - capacity_ah / spec.q0_ah
    return spec.r0_ohm * (1.0 + 0.6 * max(loss, 0.0))


def simulate_cycle(
    spec: CellSpec,
    cycle_no: int,
    *,
    total_cycles: int,
    rng: np.random.Generator,
    dt_s: float = 10.0,
    start_ts: float = 1_700_000_000.0,
    jitter: float = 0.5,
    noise_v: float = DEFAULT_NOISE_V,
    noise_i: float = DEFAULT_NOISE_I,
    noise_t: float = DEFAULT_NOISE_T,
    ambient_c: float = 25.0,
) -> tuple[dict, dict]:
    """生成一个放电循环的会话与它的真值。

    返回 ``(session, truth)``。``session`` 直接满足 Schema v1；
    ``truth`` 是 ``{"battery_id", "cycle", "capacity_ah", "soh"}``。
    """
    if dt_s <= 0:
        raise ValueError(f"dt_s 必须为正，收到 {dt_s}")
    capacity = capacity_at(spec, cycle_no, total_cycles)
    resistance = internal_resistance(spec, capacity)
    # γ 随老化略降：老电池的低 SOC 端更「软」。这一项与内阻增大一起，
    # 让 ΔQ(V) 的**形状**（不只是幅度）随 SOH 变化。
    gamma = spec.gamma * (1.0 - 0.15 * (1.0 - capacity / spec.q0_ah))

    current = -abs(spec.discharge_a)
    duration_s = capacity * 3600.0 / abs(current)

    # 名义等间隔采样 + 每步乘一个 [1−jitter, 1+jitter] 的随机因子。
    # 用累积和而不是逐点重采样，保证 t 严格递增。
    n_steps = max(int(np.ceil(duration_s / dt_s)), 2)
    factors = 1.0 + rng.uniform(-jitter, jitter, size=n_steps)
    # ``offsets`` 是**相对放电起点**的时间，从 0 开始——物理量（已放出容量）
    # 必须用它算。落到会话里的绝对时间另外加静置段的长度。
    # 曾经把静置段直接并进 offsets，于是放电段的第一个点就带着 40 s 的容量，
    # 时间戳还先降后升（静置段占 [1·dt, 4·dt]，放电段从 0 开始）。
    offsets = np.cumsum(np.concatenate([[0.0], factors[:-1] * dt_s]))
    offsets = offsets * (duration_s / offsets[-1])  # 收尾对齐到满容量
    # 静置段占 [1·dt, REST_POINTS·dt]，所以放电从 (REST_POINTS+1)·dt 起算。
    # 早先写成 arange(REST_POINTS, 0, -1)（从 4·dt 递减到 1·dt），时间戳
    # 于是先降后升；而分段与积分都不看时间单调性，只有 Schema v1 会拦。
    t_discharge = offsets + (REST_POINTS + 1) * dt_s

    discharged_ah = abs(current) * offsets / 3600.0
    soc = np.clip(1.0 - discharged_ah / capacity, 0.0, 1.0)
    voltage = open_circuit_voltage(soc, gamma=gamma) + current * resistance
    voltage = voltage + rng.normal(0.0, noise_v, size=n_steps)
    current_trace = np.full(n_steps, current) + rng.normal(0.0, noise_i, size=n_steps)
    # 温度：放电过程中缓慢上升，幅度与放出容量成正比
    temperature = (
        ambient_c
        + 4.0 * (discharged_ah / max(capacity, 1e-9))
        + rng.normal(0.0, noise_t, size=n_steps)
    )
    soc_est = np.clip(soc + rng.normal(0.0, 0.005, size=n_steps), 0.0, 1.0)

    # 静置段：前后各一小段 I=0。电压取该 SOC 下的开路电压。
    rest_dt = dt_s
    pre_t = np.arange(1, REST_POINTS + 1) * rest_dt
    post_t = t_discharge[-1] + np.arange(1, REST_POINTS + 1) * rest_dt
    pre_v = np.full(REST_POINTS, float(open_circuit_voltage(1.0, gamma=gamma))) + rng.normal(
        0.0, noise_v, size=REST_POINTS
    )
    post_v = np.full(REST_POINTS, float(open_circuit_voltage(0.0, gamma=gamma))) + rng.normal(
        0.0, noise_v, size=REST_POINTS
    )

    t_rel = np.concatenate([pre_t, t_discharge, post_t])
    v_all = np.concatenate([pre_v, voltage, post_v])
    i_all = np.concatenate([np.zeros(REST_POINTS), current_trace, np.zeros(REST_POINTS)])
    # 静置段温度取环境温度。**不能填 0**：0 °C 是合法值（schema 允许 −30…100），
    # 所以它不会报错，只会让温度通道在每段首尾出现一个 25 °C 的阶跃。
    temp_all = np.concatenate(
        [
            np.full(REST_POINTS, ambient_c),
            temperature,
            np.full(REST_POINTS, float(temperature[-1])),
        ]
    )
    soc_all = np.concatenate([np.ones(REST_POINTS), soc_est, np.zeros(REST_POINTS)])

    records = [
        {
            "t": round(float(start_ts + t_rel[k]), 3),
            "V": round(float(v_all[k]), 6),
            "I": round(float(i_all[k]), 6),
            "T": round(float(temp_all[k]), 4),
            "soc_est": round(float(soc_all[k]), 6),
        }
        for k in range(len(t_rel))
    ]

    session = {
        "schema_version": "v1",
        "user_id_hash": hashlib.sha256(spec.battery_id.encode("utf-8")).hexdigest()[:32],
        "battery_id": spec.battery_id,
        "session_id": f"{spec.battery_id}-c{cycle_no:04d}",
        "start_ts": records[0]["t"],
        "end_ts": records[-1]["t"],
        "chemistry_hint": "LiFePO4",
        "sampling_hint": "variable",
        "records": records,
    }
    truth = {
        "battery_id": spec.battery_id,
        "session_id": session["session_id"],
        "cycle": int(cycle_no),
        "capacity_ah": float(capacity),
    }
    return session, truth


def simulate_battery(
    spec: CellSpec,
    *,
    n_cycles: int,
    seed: int = 0,
    first_cycle: int = 1,
    dt_s: float = 10.0,
    start_ts: float = 1_700_000_000.0,
    cycle_gap_s: float = 3600.0,
    **kwargs,
) -> tuple[list[dict], list[dict]]:
    """一颗电池跑 ``n_cycles`` 个循环，返回 ``(sessions, truths)``。

    ``total_cycles`` 传 ``n_cycles``，所以最后一个循环的 SOH 恰好是
    ``1 − fade``——真值轨迹的端点因此可预测，测试可以直接断言。
    """
    if n_cycles <= 0:
        raise ValueError(f"n_cycles 必须为正，收到 {n_cycles}")
    rng = np.random.default_rng(seed)
    sessions, truths = [], []
    for k in range(n_cycles):
        cycle_no = first_cycle + k
        session, truth = simulate_cycle(
            spec,
            cycle_no,
            total_cycles=first_cycle + n_cycles - 1,
            rng=rng,
            dt_s=dt_s,
            start_ts=start_ts + k * cycle_gap_s,
            **kwargs,
        )
        sessions.append(session)
        truths.append(truth)
    # SOH 相对**首个循环**，不是相对出厂容量——实验室路径的定义是
    # ``SOH_k = QD_k / QD_ref``（ref = 首个循环），两条路径的标签必须同尺度，
    # 否则联调出来的 MAE 与 M1–M4 的数不能放在一起看。
    q_ref = truths[0]["capacity_ah"]
    for truth in truths:
        truth["soh"] = float(truth["capacity_ah"] / q_ref)
    return sessions, truths


def simulate_fleet(
    n_cells: int,
    *,
    n_cycles: int = 20,
    seed: int = 0,
    q0_range: tuple[float, float] = (0.95, 1.25),
    fade_range: tuple[float, float] = (0.06, 0.18),
    **kwargs,
) -> tuple[list[dict], list[dict]]:
    """一批电池。``q0`` 与 ``fade`` 在给定区间里均匀抽——**电池间差异必须存在**，
    否则所有电池的 SOH 轨迹完全重合，客户端之间没有异质性可谈。
    """
    if n_cells <= 0:
        raise ValueError(f"n_cells 必须为正，收到 {n_cells}")
    rng = np.random.default_rng(seed)
    sessions, truths = [], []
    for i in range(n_cells):
        spec = CellSpec(
            battery_id=f"sim{i:03d}",
            q0_ah=float(rng.uniform(*q0_range)),
            fade=float(rng.uniform(*fade_range)),
            r0_ohm=float(rng.uniform(0.025, 0.040)),
        )
        s, t = simulate_battery(
            spec, n_cycles=n_cycles, seed=int(rng.integers(0, 2**31 - 1)), **kwargs
        )
        sessions.extend(s)
        truths.extend(t)
    return sessions, truths


def write_sessions_jsonl(sessions: list[dict], path: str | Path) -> Path:
    """一行一个会话。UTF-8、无 BOM、``\\n`` 换行——JSONL 的消费者是流式读取的，
    CRLF 会在某些解析器里留下尾部 ``\\r`` 并让最后一条记录解析失败。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for session in sessions:
            fh.write(json.dumps(session, ensure_ascii=False, separators=(",", ":")))
            fh.write("\n")
    return path


def write_truth_json(truths: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"n_sessions": len(truths), "sessions": truths}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path
