#!/usr/bin/env python3
"""从 MATR 原始曲线生成 per-cycle 特征表（parquet + spec）。

特征口径
--------
每行 = (电池, 循环 k)，特征 = ``Qc_k(V) − Qc_ref(V)`` 在 1000 点电压网格上的
差分，标签 = ``SOH_k = QD_k / QD_ref``。

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
    SOH_COL,
    DataContractError,
    delta_q_features,
    make_v_grid,
    validate_feature_table,
)
from flsoh.matr import MatrBatch, MatrError  # noqa: E402

DEFAULT_CYCLES = "5:100:5"
DEFAULT_V_LO = 2.0
DEFAULT_V_HI = 3.5
DEFAULT_POINTS = 1000
DEFAULT_REF_CYCLE = 1
DEFAULT_WORKERS = 4

STATE_NAME = "state.json"


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
    """
    return {
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
    """抽一颗电池的所有行。读不出/覆盖不住的循环记进 ``drops``，不抛异常。"""
    result = CellResult(cell_id=cell_id)
    try:
        summary = batch.summary(index, fields=("QDischarge", "cycle"))
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
        v_ref, c_ref = batch.charge_curve(index, ref_cycle)
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
            voltage, capacity = batch.charge_curve(index, cycle)
            feature = delta_q_features(
                [(v_ref, c_ref), (voltage, capacity)], grid, reference_index=0
            )[0]
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
    try:
        batch.prefetch_cell(index, [ref_cycle, *cycles])
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
) -> list[CellResult]:
    """一个 worker 抽一小批电池，**自己开 h5py 句柄**。

    h5py 不是线程安全的，共用句柄并发读会静默错位，而错位后的指标完全正常。
    所以宁可每个 worker 多付一次 HDF5 元数据的往返，也不共用句柄。
    """
    with MatrBatch(source, batch_no=batch_no) as batch:
        return [
            _extract_one(batch, meta, cycles=cycles, ref_cycle=ref_cycle, grid=grid)
            for meta in metas
        ]


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
    """
    metas = list(entry["cells"] if todo is None else todo)
    if limit is not None:
        metas = metas[:limit]
    if not metas:
        return []
    workers = max(1, min(int(workers), len(metas)))
    if workers == 1:
        return _extract_chunk(
            source,
            int(entry["batch_no"]),
            metas,
            cycles=cycles,
            ref_cycle=ref_cycle,
            grid=grid,
        )
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
        for future in futures:
            out.extend(future.result())
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
        help="批次内并发的 h5py 句柄数",
    )
    parser.add_argument("--limit-cells", type=int, default=None, help="每批只跑前 N 颗")
    parser.add_argument(
        "--restart", action="store_true", help="清空已有分片重来（规格变了才会用到）"
    )
    parser.add_argument(
        "--keep-parts", action="store_true", help="跑完保留分片目录（默认删除）"
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
        if not (parts / f"{meta['cell_id']}.parquet").is_file()
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

    lock = threading.Lock()
    done = [total - len(todo)]
    started = time.time()

    def report(results: list[CellResult], rows: list[int], seconds: float) -> None:
        with lock:
            for result, n_rows in zip(results, rows):
                done[0] += 1
                print(
                    f"  [{done[0]:>3d}/{total}] {result.cell_id:<7s} "
                    f"{n_rows:>3d} 行  {seconds:>6.1f}s"
                    + (f"  丢弃 {len(result.drops)}" if result.drops else ""),
                    flush=True,
                )

    all_drops: list[dict] = []
    for entry in entries:
        batch_no = int(entry["batch_no"])
        batch_todo = [meta for ent, meta in todo if ent is entry]
        if not batch_todo:
            continue
        print(f"[batch {batch_no}] {len(batch_todo)} 颗待抽", flush=True)
        source = sources[batch_no]
        workers = max(1, args.workers)
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
                ): chunk
                for chunk in chunks
            }
            for future in as_completed(futures):
                t0 = time.time()
                chunk_results = future.result()
                rows: list[int] = []
                for result in chunk_results:
                    frame = cell_frame(result, args.points)
                    if frame is None:
                        # 一颗都没抽到时也留痕：没有分片 ≠ 没跑过，
                        # 否则下次还会再跑一遍这颗，永远看不到它失败在哪。
                        (parts / f"{result.cell_id}.empty").write_text(
                            json.dumps(result.drops, ensure_ascii=False, indent=2)
                            + "\n",
                            encoding="utf-8",
                        )
                        rows.append(0)
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
                        rows.append(len(frame))
                    all_drops.extend(
                        {"cell_id": result.cell_id, **drop} for drop in result.drops
                    )
                report(chunk_results, rows, time.time() - t0)

    # 这一轮之前跑掉的那些，也要把丢弃记录补进来。
    for entry in entries:
        for meta in entry["cells"][: args.limit_cells if args.limit_cells else None]:
            empty = parts / f"{meta['cell_id']}.empty"
            if empty.is_file():
                try:
                    for drop in json.loads(empty.read_text(encoding="utf-8")):
                        all_drops.append({"cell_id": str(meta["cell_id"]), **drop})
                except (OSError, json.JSONDecodeError):
                    pass

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
        "feature": "ΔQ(V) = Qc_k(V) − Qc_ref(V)，只用充电片段",
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
    if args.keep_parts:
        print(f"分片保留在 {parts}")
    else:
        shutil.rmtree(parts, ignore_errors=True)
        print(f"已清理 {parts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
