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

用法
----
    python scripts/build_features.py --manifest data/mit-stanford/manifest.json \\
        --out data/mit-stanford/features.parquet

    # 快速验证：只跑 3 颗
    python scripts/build_features.py --manifest ... --out /tmp/f.parquet \\
        --limit-cells 3
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flsoh.data import (  # noqa: E402
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


def extract_batch(
    entry: dict,
    source: str,
    *,
    cycles: list[int],
    ref_cycle: int,
    grid: np.ndarray,
    limit: int | None = None,
) -> list[CellResult]:
    cells = entry["cells"][: limit if limit else None]
    results: list[CellResult] = []
    with MatrBatch(source, batch_no=entry["batch_no"]) as batch:
        for meta in cells:
            index = int(meta["index"])
            # 先按字节区间把这一颗要用的曲线并发抓回，再逐条解析。
            # 不预取的话每条曲线两次往返，20 条就是 40 次。
            try:
                batch.prefetch_cell(index, [ref_cycle, *cycles])
            except MatrError:
                pass  # 预取失败不影响正确性，退化为逐条读
            results.append(
                extract_cell(
                    batch,
                    str(meta["cell_id"]),
                    index,
                    cycles=cycles,
                    ref_cycle=ref_cycle,
                    grid=grid,
                )
            )
    return results


def to_frame(results: list[CellResult], n_points: int) -> pd.DataFrame:
    cell_ids: list[str] = []
    cycle_values: list[int] = []
    soh_values: list[float] = []
    blocks: list[np.ndarray] = []
    for result in results:
        if not result.rows:
            continue
        cell_ids.extend([result.cell_id] * len(result.rows))
        cycle_values.extend(result.cycles)
        soh_values.extend(result.soh)
        blocks.append(np.stack(result.rows))
    if not blocks:
        raise DataContractError("一行特征都没抽到；先看 dropped 里的原因")

    features = np.concatenate(blocks, axis=0).astype(np.float32)
    frame = pd.DataFrame(features, columns=[f"f{i}" for i in range(n_points)])
    frame.insert(0, CELL_COL, cell_ids)
    frame.insert(1, CYCLE_COL, np.asarray(cycle_values, dtype=np.int32))
    frame.insert(2, SOH_COL, np.asarray(soh_values, dtype=np.float32))
    validate_feature_table(frame)
    return frame


def summarise_drops(results: list[CellResult]) -> dict:
    """把逐条 drop 聚合成「原因 → 条数」，spec 里只放聚合结果。"""
    counts: dict[str, int] = {}
    per_cell: dict[str, int] = {}
    for result in results:
        if result.drops:
            per_cell[result.cell_id] = len(result.drops)
        for drop in result.drops:
            reason = drop["reason"]
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
    parser.add_argument("--workers", type=int, default=3, help="按批次并发")
    parser.add_argument("--limit-cells", type=int, default=None, help="每批只跑前 N 颗")
    args = parser.parse_args(argv)

    from build_manifest import resolve_source

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cycles = parse_cycles(args.cycles)
    grid = make_v_grid(args.v_lo, args.v_hi, args.points)

    entries = manifest["batches"]
    sources = [
        resolve_source(entry["batch_no"], args.source_dir) for entry in entries
    ]
    print(f"批次 {[e['batch_no'] for e in entries]}，循环 {cycles[0]}..{cycles[-1]} "
          f"共 {len(cycles)} 个，网格 {args.points} 点 [{args.v_lo}, {args.v_hi}] V",
          flush=True)

    def run(pair):
        entry, source = pair
        print(f"[batch {entry['batch_no']}] 开始", flush=True)
        out = extract_batch(
            entry,
            source,
            cycles=cycles,
            ref_cycle=args.ref_cycle,
            grid=grid,
            limit=args.limit_cells,
        )
        print(
            f"[batch {entry['batch_no']}] {len(out)} 颗，"
            f"{sum(len(r.rows) for r in out)} 行",
            flush=True,
        )
        return out

    results: list[CellResult] = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, len(entries)))) as pool:
        for part in pool.map(run, list(zip(entries, sources))):
            results.extend(part)

    frame = to_frame(results, args.points)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)

    drops = summarise_drops(results)
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
        "dropped": drops,
    }
    spec_path = args.out.with_suffix(".spec.json")
    spec_path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print(f"行数 {spec['rows']}，电池 {spec['cells_with_rows']} 颗，"
          f"特征 {args.points} 维")
    if drops["by_reason"]:
        print("丢弃的循环：")
        for reason, count in sorted(drops["by_reason"].items(), key=lambda kv: -kv[1]):
            print(f"  {count:>5} 条  {reason}")
    print(f"写入 {args.out}")
    print(f"规格 {spec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
