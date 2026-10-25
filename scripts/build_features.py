#!/usr/bin/env python3
"""从 MATR 原始曲线生成 per-cycle 特征表（parquet + spec）。

特征口径
--------
每行 = (电池, 循环 k)，特征 = ``Qd_k(V) − Qd_ref(V)`` 在 1000 点电压网格上的
差分，标签 = ``SOH_k = QD_k / QD_ref``。

**为什么是放电段（``Qd``）而不是充电段（``Qc``）**：恒压充电段里电压几乎不变
而容量猛涨，dQ/dV 趋于无穷，均匀电压网格在那里会把 0.1 Ah 的容量变化压进
一两个网格点。实测同一颗电池同一对循环，插值出的最大相邻跳变：充电段
**1.118 Ah**，放电段 **0.013 Ah**。这不是调参能解决的，是充电曲线在恒定电压
下的固有奇性。Severson et al. 2019 用的也是放电曲线。

顺带一段踩过的坑（2026-10-25）：曾经把**整条循环**（充电 + 放电拼在一起）
喂给 ``delta_q_curve``，它的 ``argsort`` 于是把放电段排进充电段的电压区间，
而放电段的 ``Qc`` 是常数——低电压区冒出一片「容量 = 满容量」的平台。那样算出
的「ΔQ」总变差 269.6，正确分段后 0.21，两者相关系数 −0.02。指标照样算得
出来，曲线形状也「看着像那么回事」，所以只有总变差这个量能拦住它。见
``scripts/check_features.py``。

三个为什么
----------
**为什么是 per-cycle**：按电池划分后每客户端只有 5 颗电池（k=5）。若每颗电池
只出一行，客户端本地只有 5 个样本，联邦训练本身不成立。per-cycle 让每客户端
有约 100 行，H1–H3 才谈得上统计。

**为什么参考循环固定**：参考循环固定为 ``--ref-cycle``（默认 1）。若每行拿
前一循环作基准，第 k 行与第 k+1 行的尺子不同，模型学到的是尺子变化而不是老化。

**为什么电压窗口是常量**：从数据算出来的窗口会随数据变，两次运行的网格不同，
特征就不可比。所以窗口写死并在 spec 里记录；覆盖不住的曲线**逐条记入
``dropped``**，不静默跳过——静默跳过会让「样本少了 30%」这件事在精度曲线上
完全看不出来。

为什么要断点续跑
----------------
135 颗 × 21 个循环 × 2 条曲线，每颗几次往返，而远端读一次要 1.4 s 到几十秒。
**这个命令不可能一次跑完**：中断一次，全部作废重来，而重来还是跑不完。

所以每抽完一颗就落一块 parquet 分片，重启时跳过已有分片。三处不能省：

1. **分片要原子落盘**。先写临时文件再 ``os.replace``——进程在写一半被 kill，
   留下的是能打开的残片，下一次就会被当成有效数据跳过。半块分片比没有分片更糟。
2. **规格指纹不匹配就拒绝续跑**。分片目录里存一份 ``state.json``，记下循环区间、
   参考循环、电压网格。改了网格接着跑，新旧行的第 i 列不是同一个物理量，
   而拼出来的表**看不出任何异常**。这时候必须停下来，不能自己判断哪个对。
3. **写完要读回来验**。核对 cell_id、行数、列数。对不上就删掉分片并报错，
   让下一次重新抽这颗。

用法
----
    python scripts/build_features.py --manifest data/mit-stanford/manifest.json \\
        --out data/mit-stanford/features.parquet

    # 中断后接着跑：同一条命令即可，已有分片会跳过
    # （前提是上一次没跑完——跑完默认清掉分片目录；要留着得加 --keep-parts）
    python scripts/build_features.py --manifest ... --out data/mit-stanford/features.parquet

    # 换了网格/循环区间：拒绝续跑，加 --restart 才清空重来
    python scripts/build_features.py --manifest ... --out ... --points 500 --restart

    # 快速验证：只跑前 3 颗
    python scripts/build_features.py --manifest ... --out /tmp/f.parquet --limit-cells 3

产出
----
    <out>              最终特征表（所有分片拼接，按 cell_id + cycle 排序）
    <out>.spec.json    网格、循环区间、丢弃统计
    <out>.parts/       分片目录，跑完默认删掉；--keep-parts 可保留
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 刻意从 flsoh.features 而不是 flsoh.data 导入：后者在模块层 import torch，
# 而本脚本要能在没装 torch 的 CI job 里跑。纯特征数学归 features。
from flsoh.features import (  # noqa: E402
    CELL_COL,
    CYCLE_COL,
    FEATURE_SPEC_VERSION,
    SOH_COL,
    DataContractError,
    delta_q_features,
    make_v_grid,
    validate_feature_table,
)
from check_features import check_feature_table  # noqa: E402
from flsoh.matr import MatrBatch, MatrError, MatrTransportError  # noqa: E402

DEFAULT_CYCLES = "5:100:5"
DEFAULT_V_LO = 2.0
DEFAULT_V_HI = 3.5
DEFAULT_POINTS = 1000
DEFAULT_REF_CYCLE = 1
DEFAULT_WORKERS = 1  # 见 --workers 的 help：服务端按聚合速率节流，调大只有 1.14 倍

STATE_NAME = "state.json"

# 传输失败的指纹。这些词只会出现在网络错误里，不会出现在数据问题的描述里
# （数据问题说的是「越界」「非正」「覆盖不住」）。用来识别旧版本写坏的
# .empty 标记——它们把一次 DNS 失败写成了「这颗电池没有数据」。
_TRANSPORT_MARKERS = (
    "gaierror",
    "getaddrinfo",
    "ConnectionReset",
    "ConnectionRefused",
    "RemoteDisconnected",
    " timed out",
    "timeout",
    "HTTP 5",
    "区间抓取失败",
    "Content-Range",
)


def _looks_like_transport(reason: str) -> bool:
    """这条丢弃原因其实是网络问题，不是数据问题。"""
    return any(marker in reason for marker in _TRANSPORT_MARKERS)


# ---------------------------------------------------------------------------
# 断点续跑


def parts_dir_for(out: Path) -> Path:
    """分片目录。紧跟输出文件，不另起一处——散落两处就容易只删一边。"""
    return out.with_name(out.name + ".parts")


def state_fingerprint(
    *,
    cycles: list[int],
    ref_cycle: int,
    v_lo: float,
    v_hi: float,
    points: int,
) -> dict:
    """这次抽取的规格指纹。

    任何一个字段不同，新旧分片拼在一起都是错的——而且错得不报异常。

    ``feature_version`` 是**怎么算**的指纹（见 ``flsoh.features``）：只改参数
    不改算法时它不变，只改算法不改参数时**只有它变**。少了它，修完算法重跑
    同一条命令会把旧分片全部当成有效数据跳过，退出码还是 0。
    """
    return {
        "feature_version": FEATURE_SPEC_VERSION,
        "cycles": list(cycles),
        "ref_cycle": int(ref_cycle),
        "v_lo": float(v_lo),
        "v_hi": float(v_hi),
        "points": int(points),
    }


def _normalise(value):
    """把 list / tuple 归一成 list 再比。

    否则哪天有人把 ``cycles`` 从 list 改成 tuple，整条命令就会因为一个无关紧要的
    类型差异拒绝续跑——而这种拒绝看起来跟「规格真的变了」一模一样。
    """
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    return value


def state_mismatch(saved: dict | None, current: dict) -> str | None:
    """返回 None 表示可以续跑，否则返回一句人能读懂的原因。

    只比字段不比 dict 相等：多一个字段不该阻止续跑（比如以后加了新字段，
    旧分片目录没有它，但旧分片仍然有效）。
    """
    if saved is None:
        return None
    for key, value in current.items():
        if _normalise(saved.get(key)) != _normalise(value):
            return (
                f"{key} 变了：已跑的是 {saved.get(key)!r}，这次是 {value!r}。"
                "改了网格或循环区间就不能接着跑，新旧行的同一列不是同一个物理量。"
                "确认要重来就加 --restart"
            )
    return None


def write_cell_part(parts_dir: Path, cell_id: str, frame: pd.DataFrame) -> Path:
    """原子落一块分片，然后读回来验。

    ``os.replace`` 在同一文件系统上是原子的，所以进程被 kill 时只会留下
    「完整」或「没有」，不会留下半块。半块 parquet 恰好还能被打开——
    那比没有更危险。
    """
    parts_dir.mkdir(parents=True, exist_ok=True)
    final = parts_dir / f"{cell_id}.parquet"
    tmp = final.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, final)
    return final


def read_cell_part(
    path: Path, *, cell_id: str, n_points: int, n_cycles: int
) -> pd.DataFrame | None:
    """读回并核对。返回 None 表示这块分片不可信（调用方应当重抽）。"""
    try:
        frame = pd.read_parquet(path)
    except Exception:  # noqa: BLE001 —— 任何读不了的理由都当不可信
        return None
    if list(frame.columns) != [CELL_COL, CYCLE_COL, SOH_COL] + [
        f"f{i}" for i in range(n_points)
    ]:
        return None
    if frame.empty or (frame[CELL_COL] != cell_id).any():
        return None
    if len(frame) != n_cycles or frame[CYCLE_COL].duplicated().any():
        return None
    return frame


def load_state(parts_dir: Path) -> dict | None:
    path = parts_dir / STATE_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_state(parts_dir: Path, state: dict) -> None:
    """state.json 也原子写——它比任何分片都关键，坏了就等于没有指纹。"""
    parts_dir.mkdir(parents=True, exist_ok=True)
    path = parts_dir / STATE_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def cell_frame(result: CellResult, n_points: int) -> pd.DataFrame | None:
    """一颗电池的结果转成表。一行都没有时返回 None——那不是错误，是抽空了。"""
    if not result.rows:
        return None
    frame = pd.DataFrame(
        np.stack(result.rows).astype(np.float32),
        columns=[f"f{i}" for i in range(n_points)],
    )
    frame.insert(0, CELL_COL, [result.cell_id] * len(result.rows))
    frame.insert(1, CYCLE_COL, np.asarray(result.cycles, dtype=np.int32))
    frame.insert(2, SOH_COL, np.asarray(result.soh, dtype=np.float32))
    return frame


def concat_parts(parts: list[Path], n_points: int) -> pd.DataFrame:
    """把分片拼成最终表。按 cell_id + cycle 排序，保证可复现。

    排序不是洁癖：分片是并发写的，落盘顺序不稳定，不排序的话两次运行
    的行序不同，下游任何按位置取行的比较都会对不上。
    """
    frames = [pd.read_parquet(path) for path in parts]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values([CELL_COL, CYCLE_COL], kind="stable").reset_index(
        drop=True
    )
    validate_feature_table(frame)
    return frame


def parse_cycles(spec: str) -> list[int]:
    """``lo:hi:step``（含两端）→ 循环下标列表。"""
    parts = spec.split(":")
    if len(parts) != 3:
        raise SystemExit(f"循环区间要写成 lo:hi:step，收到 {spec!r}")
    lo, hi, step = (int(x) for x in parts)
    if lo < 1 or hi < lo or step < 1:
        raise SystemExit(f"循环区间不合法：{spec}")
    return list(range(lo, hi + 1, step))


@dataclass
class CellResult:
    cell_id: str
    cycles: list[int] = field(default_factory=list)
    soh: list[float] = field(default_factory=list)
    rows: list[np.ndarray] = field(default_factory=list)
    drops: list[dict] = field(default_factory=list)


def extract_cell(
    batch: MatrBatch,
    cell_id: str,
    index: int,
    *,
    cycles: list[int],
    ref_cycle: int,
    grid: np.ndarray,
) -> CellResult:
    """抽一颗电池的所有行。

    只有**数据本身的问题**才记进 ``drops``（循环越界、容量非正、曲线覆盖不住
    电压窗口）。传输错误往上抛——它不是数据缺失，记进 drops 等于把一次网络
    抖动写成一条数据缺陷：spec 里会躺着几十条「参考曲线读不出」，
    读起来就是"这些电池没有数据"。
    """
    result = CellResult(cell_id=cell_id)
    try:
        summary = batch.summary(index, fields=("QDischarge", "cycle"))
    except MatrTransportError:
        raise
    except MatrError as exc:
        result.drops.append({"cycle": None, "reason": f"summary 读不出：{exc}"})
        return result

    discharge = summary["QDischarge"]
    n_cycles = len(discharge)
    if not 0 <= ref_cycle < n_cycles:
        result.drops.append(
            {"cycle": None, "reason": f"参考循环 {ref_cycle} 越界（共 {n_cycles} 个）"}
        )
        return result
    q_ref = float(discharge[ref_cycle])
    if not np.isfinite(q_ref) or q_ref <= 0:
        result.drops.append(
            {"cycle": None, "reason": f"参考循环容量非正：{q_ref:.4f}"}
        )
        return result

    try:
        v_ref, c_ref = batch.discharge_curve(index, ref_cycle)
    except MatrTransportError:
        raise
    except MatrError as exc:
        result.drops.append({"cycle": None, "reason": f"参考曲线读不出：{exc}"})
        return result

    for cycle in cycles:
        if cycle >= n_cycles:
            result.drops.append(
                {"cycle": cycle, "reason": f"该电池只有 {n_cycles} 个循环"}
            )
            continue
        try:
            voltage, capacity = batch.discharge_curve(index, cycle)
            feature = delta_q_features(
                [(v_ref, c_ref), (voltage, capacity)], grid, reference_index=0
            )[0]
        except MatrTransportError:
            raise
        except (DataContractError, MatrError) as exc:
            result.drops.append({"cycle": cycle, "reason": str(exc)})
            continue
        result.cycles.append(cycle)
        result.soh.append(float(discharge[cycle]) / q_ref)
        result.rows.append(feature)
    return result


def _extract_one(
    batch: MatrBatch,
    meta: dict,
    *,
    cycles: list[int],
    ref_cycle: int,
    grid: np.ndarray,
) -> CellResult:
    index = int(meta["index"])
    # 先按字节区间把这一颗要用的曲线并发抓回，再逐条解析。
    # 不预取的话每条曲线两次往返，20 条就是 40 次。
    # 三个通道都要：V 与 Qd 是曲线本身，I 用来切放电段（2026-10-25 的坑）。
    try:
        batch.prefetch_cell(index, [ref_cycle, *cycles], fields=("V", "Qd", "I"))
    except MatrTransportError:
        raise  # 预取阶段的网络失败同样不是数据问题，不能吞
    except MatrError:
        pass  # 预取失败不影响正确性，退化为逐条读
    return extract_cell(
        batch,
        str(meta["cell_id"]),
        index,
        cycles=cycles,
        ref_cycle=ref_cycle,
        grid=grid,
    )


def _extract_chunk(
    source: str,
    batch_no: int,
    metas: list[dict],
    *,
    cycles: list[int],
    ref_cycle: int,
    grid: np.ndarray,
    on_result=None,
) -> tuple[list[CellResult], MatrTransportError | None]:
    """一个 worker 抽一小批电池，**自己开 h5py 句柄**。

    h5py 不是线程安全的，共用句柄并发读会静默错位，而错位后的指标完全正常。
    所以宁可每个 worker 多付一次 HDF5 元数据的往返，也不共用句柄。

    ``on_result`` 每抽完一颗就调一次。**这不是便利设施而是必需的**：
    分片必须在抽到的瞬间落盘。若等整块（默认 8 颗）跑完再写，
    中断一次就丢掉 8 颗 × 约 60 s 的成果，而这一轮本来就要跑一小时。

    返回 ``(已抽到的结果, 传输错误或 None)``。传输错误不往上抛同样是为了
    **不丢已抽到的那几颗**——先把成果通过 ``on_result`` 交出去落盘，
    再让调用方决定中止。
    """
    results: list[CellResult] = []
    error: MatrTransportError | None = None
    try:
        with MatrBatch(source, batch_no=batch_no) as batch:
            for meta in metas:
                result = _extract_one(
                    batch, meta, cycles=cycles, ref_cycle=ref_cycle, grid=grid
                )
                results.append(result)
                if on_result is not None:
                    on_result(result)
    except MatrTransportError as exc:
        error = exc
    return results, error


def extract_batch(
    entry: dict,
    source: str,
    *,
    cycles: list[int],
    ref_cycle: int,
    grid: np.ndarray,
    limit: int | None = None,
    workers: int = 1,
    todo: list[dict] | None = None,
) -> list[CellResult]:
    """抽一个批次。

    ``todo`` 给了就只抽这些（续跑时用），否则抽 ``entry['cells']``。
    ``workers`` 是**批次内**的并发句柄数——之前并发只发生在批次之间，
    而批次只有 3 个，等于大部分时间在串行等往返。

    传输错误在这里**往上抛**：本函数是给测试和一次性调用用的，
    它们没有分片目录可以续跑，所以必须让失败可见。
    """
    metas = list(entry["cells"] if todo is None else todo)
    if limit is not None:
        metas = metas[:limit]
    if not metas:
        return []
    workers = max(1, min(int(workers), len(metas)))
    if workers == 1:
        results, error = _extract_chunk(
            source,
            int(entry["batch_no"]),
            metas,
            cycles=cycles,
            ref_cycle=ref_cycle,
            grid=grid,
        )
        if error is not None:
            raise error
        return results
    size = -(-len(metas) // workers)
    chunks = [metas[i : i + size] for i in range(0, len(metas), size)]
    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        futures = [
            pool.submit(
                _extract_chunk,
                source,
                int(entry["batch_no"]),
                chunk,
                cycles=cycles,
                ref_cycle=ref_cycle,
                grid=grid,
            )
            for chunk in chunks
        ]
        out: list[CellResult] = []
        errors: list[MatrTransportError] = []
        for future in futures:
            results, error = future.result()
            out.extend(results)
            if error is not None:
                errors.append(error)
        if errors:
            raise errors[0]
    return out


def to_frame(results: list[CellResult], n_points: int) -> pd.DataFrame:
    frames = [cell_frame(result, n_points) for result in results]
    frames = [frame for frame in frames if frame is not None]
    if not frames:
        raise DataContractError("一行特征都没抽到；先看 dropped 里的原因")
    frame = pd.concat(frames, ignore_index=True)
    validate_feature_table(frame)
    return frame


def summarise_drops(records: list[dict]) -> dict:
    """把逐条 drop 聚合成「原因 → 条数」。spec 里只放聚合结果。

    ``records`` 每条形如 ``{"cell_id": ..., "cycle": ..., "reason": ...}``——
    续跑时要能跨多次运行累加，所以必须是扁平的记录，不能是某一次的
    ``CellResult`` 列表。
    """
    counts: dict[str, int] = {}
    per_cell: dict[str, int] = {}
    for record in records:
        per_cell[record["cell_id"]] = per_cell.get(record["cell_id"], 0) + 1
        reason = record["reason"]
        # 把具体循环号抹掉再聚合，否则每条的 key 都不同
        key = reason.split("（")[0][:60]
        counts[key] = counts.get(key, 0) + 1
    return {"by_reason": counts, "per_cell": per_cell}


def cell_is_done(parts: Path, cell_id: str) -> bool:
    """这颗算跑完了吗？

    只有 ``.parquet`` 和**真数据问题**的 ``.empty`` 算跑完。传输失败留下的
    ``.empty`` 不算——那是旧版本把网络抖动写成了数据缺失，必须重抽。

    为什么不干脆让所有 ``.empty`` 都算跑完：那正是最初那次真实故障的形态。
    DNS 抽了一次，46 颗全落下 ``.empty``，若它算"跑完"，这颗就永远跳过，
    特征表少掉三分之一。所以默认必须是"没跑完"，只有证据表明是数据问题时
    才敢跳过。
    """
    if (parts / f"{cell_id}.parquet").is_file():
        return True
    empty = parts / f"{cell_id}.empty"
    if not empty.is_file():
        return False
    try:
        saved = json.loads(empty.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False  # 读不出来的标记不信任，重抽
    return bool(saved) and not any(
        _looks_like_transport(record.get("reason", "")) for record in saved
    )


def heal_stale_empty(parts: Path, cells: list[dict]) -> list[str]:
    """删掉传输失败留下的 ``.empty``，返回被自愈的 cell_id 列表。

    旧版本（``MatrError`` 还没分类时）会把 ``getaddrinfo failed`` 记成
    「参考曲线读不出」，于是 spec 的 ``dropped`` 里躺着 46 条假的数据缺失。
    标记删掉之后，这些电池重新排队，审计也恢复诚实。
    """
    healed: list[str] = []
    for meta in cells:
        empty = parts / f"{meta['cell_id']}.empty"
        if not empty.is_file():
            continue
        try:
            saved = json.loads(empty.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if any(_looks_like_transport(record.get("reason", "")) for record in saved):
            empty.unlink()
            healed.append(str(meta["cell_id"]))
    return healed


def resume_hint(restart: bool) -> str:
    """网络中断后的续跑提示。

    ``--restart`` 在 ``main`` 开头就 ``rmtree(parts)``，所以「重跑同一条命令」
    在带 ``--restart`` 时是**错的**——照抄会把刚攒的分片全删掉再从头跑。
    这条提示写错不会报任何异常，只是让用户在 3.5 小时的任务上反复白跑，
    所以单独抽出来测。
    """
    if restart:
        return "重跑时**去掉 --restart**（它会先清空分片目录），已有的分片会跳过。"
    return "重跑同一条命令即可续跑，已有的分片会跳过。"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 MATR per-cycle 特征表")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="输出 .parquet")
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--cycles", default=DEFAULT_CYCLES, help="lo:hi:step")
    parser.add_argument("--ref-cycle", type=int, default=DEFAULT_REF_CYCLE)
    parser.add_argument("--v-lo", type=float, default=DEFAULT_V_LO)
    parser.add_argument("--v-hi", type=float, default=DEFAULT_V_HI)
    parser.add_argument("--points", type=int, default=DEFAULT_POINTS)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "批次内并发的 h5py 句柄数。默认 1，实测调大没有意义。"
            # 注意这里是全角 ％（U+FF05）而不是半角 %：argparse 会拿 help
            # 字符串做 % 格式化，半角 % 后面跟全角逗号在 Python 3.14 上直接
            # ValueError（unsupported format character）。写成 %% 在老版本
            # 上又会显示成两个百分号，所以用全角最稳。tests/test_cli_help.py
            # 钉住这件事。
            "单颗是 134 个串行小请求、抓取耗时占墙钟 98.2％，看着像「颗级"
            "并行应该有效」，但 4 颗一组实测：workers=1 单颗 120.5 s、"
            "workers=2 127.8 s、workers=4 106.1 s，只有 1.14 倍。有效聚合"
            "速率约 1.1–1.3 次/秒，开多少连接都不涨（机制未查清，见 "
            "HTTPRangeFile 的 docstring）。要提速只能减少请求或换时段——"
            "早上约 24 s/颗、晚间约 120 s/颗，差 5 倍。"
        ),
    )
    parser.add_argument("--limit-cells", type=int, default=None, help="每批只跑前 N 颗")
    parser.add_argument(
        "--restart", action="store_true", help="清空已有分片重来（规格变了才会用到）"
    )
    parser.add_argument(
        "--keep-parts", action="store_true", help="跑完保留分片目录（默认删除）"
    )
    parser.add_argument(
        "--no-check", action="store_true",
        help="跳过后写的光滑性自检（不建议：那正是 2026-10-25 那个 bug 漏网的原因）",
    )
    args = parser.parse_args(argv)

    from build_manifest import resolve_source

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cycles = parse_cycles(args.cycles)
    grid = make_v_grid(args.v_lo, args.v_hi, args.points)
    state = state_fingerprint(
        cycles=cycles,
        ref_cycle=args.ref_cycle,
        v_lo=args.v_lo,
        v_hi=args.v_hi,
        points=args.points,
    )

    parts = parts_dir_for(args.out)
    if args.restart and parts.is_dir():
        shutil.rmtree(parts)
        print(f"已清空 {parts}", flush=True)
    saved = load_state(parts)
    reason = state_mismatch(saved, state)
    if reason:
        raise SystemExit(f"拒绝续跑：{reason}")
    save_state(parts, state)

    entries = manifest["batches"]
    sources = {
        entry["batch_no"]: resolve_source(entry["batch_no"], args.source_dir)
        for entry in entries
    }

    todo = [
        (entry, meta)
        for entry in entries
        for meta in entry["cells"][: args.limit_cells if args.limit_cells else None]
        if not cell_is_done(parts, str(meta["cell_id"]))
    ]
    total = sum(
        len(entry["cells"][: args.limit_cells if args.limit_cells else None])
        for entry in entries
    )
    print(
        f"循环 {cycles[0]}..{cycles[-1]} 共 {len(cycles)} 个，"
        f"网格 {args.points} 点 [{args.v_lo}, {args.v_hi}] V，"
        f"参考循环 {args.ref_cycle}",
        flush=True,
    )
    if todo:
        print(f"待抽 {len(todo)} 颗，已有 {total - len(todo)} 颗分片", flush=True)
    else:
        print(f"{total} 颗全部有分片，直接拼接", flush=True)

    # 传输失败留下的 .empty 在这里删掉：它们不该出现在审计里，
    # 而且删晚了会让「已跑完」的计数对不上。
    for cell_id in heal_stale_empty(parts, [m for _, m in todo]):
        print(f"  {cell_id} 的 .empty 是网络失败留下的，已删除并重新排队", flush=True)

    lock = threading.Lock()
    done = [total - len(todo)]
    started = time.time()
    all_drops: list[dict] = []

    def persist(result: CellResult) -> None:
        """把一颗的结果落盘并打印。worker 线程每抽完一颗就调这个。

        **必须在 worker 里调用，不能等整块跑完**。这一轮要跑一小时，
        而中断（网络、Ctrl-C、机器休眠）随时会来；等整块 8 颗跑完再落盘，
        等于每次中断都白丢 8 × 60 s。
        """
        with lock:
            frame = cell_frame(result, args.points)
            if frame is None:
                # 一颗都没抽到时也留痕：没有分片 ≠ 没跑过，
                # 否则下次还会再跑一遍这颗，永远看不到它失败在哪。
                # 走到这里说明 drops 里全是数据问题——传输错误在
                # _extract_chunk 就被拦下，不会生成 .empty。
                (parts / f"{result.cell_id}.empty").write_text(
                    json.dumps(result.drops, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                n_rows = 0
            else:
                path = write_cell_part(parts, result.cell_id, frame)
                back = read_cell_part(
                    path,
                    cell_id=result.cell_id,
                    n_points=args.points,
                    n_cycles=len(cycles),
                )
                if back is None:
                    path.unlink(missing_ok=True)
                    raise SystemExit(
                        f"{result.cell_id} 的分片写坏并已删除，请重跑这一颗"
                    )
                n_rows = len(frame)
            all_drops.extend(
                {"cell_id": result.cell_id, **drop} for drop in result.drops
            )
            done[0] += 1
            elapsed = time.time() - started
            finished_now = done[0] - (total - len(todo))
            eta = ""
            if finished_now > 0:
                per_cell = elapsed / finished_now
                left = total - done[0]
                eta = f"  剩余 {left} 颗约 {left * per_cell / 60:.0f} 分钟"
            print(
                f"  [{done[0]:>3d}/{total}] {result.cell_id:<7s} "
                f"{n_rows:>3d} 行  {elapsed:>6.0f}s 累计{eta}"
                + (f"  丢弃 {len(result.drops)}" if result.drops else ""),
                flush=True,
            )
    for entry in entries:
        batch_no = int(entry["batch_no"])
        batch_todo = [meta for ent, meta in todo if ent is entry]
        if not batch_todo:
            continue
        print(f"[batch {batch_no}] {len(batch_todo)} 颗待抽", flush=True)
        source = sources[batch_no]
        workers = max(1, args.workers)
        # 一个 future 就是一次「打开批次」的成本（约 28 个请求，全是 HDF5
        # 元数据导航）。块越大这笔钱摊得越薄，所以块大小按「每个 worker 一块」
        # 取，而不是按固定颗数。真正决定中断损失的是 persist 回调——
        # 它每颗都落盘，所以块大小只影响元数据开销，不影响断点粒度。
        size = -(-len(batch_todo) // workers)
        chunks = [batch_todo[i : i + size] for i in range(0, len(batch_todo), size)]
        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            futures = {
                pool.submit(
                    _extract_chunk,
                    source,
                    batch_no,
                    chunk,
                    cycles=cycles,
                    ref_cycle=args.ref_cycle,
                    grid=grid,
                    on_result=persist,
                ): chunk
                for chunk in chunks
            }
            pending: list[MatrTransportError] = []
            for future in as_completed(futures):
                _, transport_error = future.result()
                if transport_error is not None:
                    pending.append(transport_error)

            if pending:
                # 传输失败不是数据缺失。这里必须中止而不是"记一笔继续跑"——
                # 继续跑会得到一张看着完整、实际少了几十颗的表，
                # 而少样本在精度曲线上完全看不出来。
                done_now = len(list(parts.glob("*.parquet"))) + len(
                    list(parts.glob("*.empty"))
                )
                print(flush=True)
                print(f"网络失败，本轮中止：{pending[0]}", flush=True)
                print(
                    f"已落盘 {done_now} 颗。{resume_hint(args.restart)}",
                    flush=True,
                )
                return 1

    # 把往届的丢弃记录补进来（真数据问题的那些 .empty）。
    for entry in entries:
        for meta in entry["cells"][: args.limit_cells if args.limit_cells else None]:
            empty = parts / f"{meta['cell_id']}.empty"
            if not empty.is_file():
                continue
            try:
                saved = json.loads(empty.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            all_drops.extend({"cell_id": str(meta["cell_id"]), **d} for d in saved)

    part_files = sorted(parts.glob("*.parquet"))
    if not part_files:
        raise SystemExit("一块分片都没有；先看上面的丢弃原因")
    frame = concat_parts(part_files, args.points)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)

    spec = {
        "generated_by": "scripts/build_features.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_manifest": str(args.manifest),
        "feature": "ΔQ(V) = Qd_k(V) − Qd_ref(V)，只用放电片段（I<0）",
        "feature_version": FEATURE_SPEC_VERSION,
        "label": "SOH_k = QD_k / QD_ref",
        "voltage_grid": {"v_lo": args.v_lo, "v_hi": args.v_hi, "n_points": args.points},
        "cycles": {
            "lo": cycles[0],
            "hi": cycles[-1],
            "step": cycles[1] - cycles[0] if len(cycles) > 1 else 1,
            "ref": args.ref_cycle,
            "count": len(cycles),
        },
        "rows": int(len(frame)),
        "cells_with_rows": int(frame[CELL_COL].nunique()),
        "cells_attempted": len(list(parts.glob("*.parquet")))
        + len(list(parts.glob("*.empty"))),
        "dropped": summarise_drops(all_drops),
    }
    spec_path = args.out.with_suffix(".spec.json")
    spec_path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print(
        f"行数 {spec['rows']}，电池 {spec['cells_with_rows']} 颗，"
        f"特征 {args.points} 维，耗时 {time.time() - started:.0f}s"
    )
    drops = spec["dropped"]
    if drops["by_reason"]:
        print("丢弃的循环：")
        for reason, count in sorted(drops["by_reason"].items(), key=lambda kv: -kv[1]):
            print(f"  {count:>5} 条  {reason}")
    print(f"写入 {args.out}")
    print(f"规格 {spec_path}")

    # 写完立刻自检：ΔQ(V) 必须是光滑的。这一步不是可选的便利设施——
    # 2026-10-25 之前没有任何检查看过特征的光滑性，于是「把整条循环喂给
    # delta_q_curve」这个错跑了整整四个里程碑，指标全部正常，只有总变差
    # 能指出数据是错的（实测 TV/幅度 86 vs 正确的 5.9）。
    if not args.no_check:
        report = check_feature_table(args.out, top=5)
        ratio = report["tv_ratio_median"]
        print(f"光滑性自检：TV/幅度 中位数 {ratio:.2f}，"
              f"最大 {report['tv_ratio_max']:.2f}，"
              f"超限 {report['n_over_limit']}/{report['n_rows']} 行，"
              f"参考循环行 {report['n_reference_rows']}（ΔQ ≡ 0，不计入比值）")
        if report["too_many_zero_rows"]:
            print(
                f"整表退化成全零：{report['n_reference_rows']} 行零而只有 "
                f"{report['n_cells']} 颗电池。TV/幅度 对全零行算不出比值，"
                "这种表会静默通过检查，所以单独拦。",
                file=sys.stderr,
            )
            return 1
        if report["n_over_limit"]:
            print(
                "特征表不够光滑——这不是精度问题，是插值链上把不相干的点"
                "配到了一起。先查分段（MATR 每个循环含充电+放电两段），"
                "再查电压网格是否落在数据值域内。",
                file=sys.stderr,
            )
            for row in report["worst"]:
                print(f"    {row['cell_id']} 循环 {row['cycle']}: "
                      f"TV/幅度 = {row['tv_ratio']:.1f}", file=sys.stderr)
            return 1
    else:
        print("光滑性自检已按 --no-check 跳过（不建议）")

    if args.keep_parts:
        print(f"分片保留在 {parts}")
    else:
        shutil.rmtree(parts, ignore_errors=True)
        print(f"已清理 {parts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
