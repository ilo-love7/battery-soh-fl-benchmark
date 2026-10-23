#!/usr/bin/env python3
"""验证「批次 2 的 5 颗是批次 1 的延续」，联网读 51 颗电池。

为什么值得单独写一个脚本
------------------------
批次 2 里有 5 颗电池是批次 1 的**延续**：同一颗电池，第一个文件跑完一段，
第二个文件接着跑。论文的 LoadData.m 列出了这个对应关系，但那只是「别人的
说法」。照抄一个会改变数据集构成的结论，风险在于：位置映射一旦猜错，删掉的
就是 5 颗正常电池，而指标上看不出来。

四条证据，全部来自数据本身
--------------------------
1. **身份**：延续段的 ``policy_readable`` 是原段 policy 的后缀。残缺是数据里的
   事实（``b1c0`` 存 14 个字符 ``3.6C(80%)-3.6C``，``b2c7`` 只存 9 个字符
   ``80%)-3.6C``），而 ``4.4C(80%)-4.4C`` 的后缀是 ``80%)-4.4C``，接不上
   ``80%)-4C``——所以这条后缀能把组唯一钉住。
2. **段长**：5 段的循环数等于论文 add_len + 1，逐个对上。5 个 add_len 互不相同，
   所以这一条同时钉住了「谁接谁」。
3. **组内 1:1**：``80%)-3.6C`` 圈出 3 颗原段、3 个延续段；``80%)-4C`` 圈出 2 颗、
   2 个延续段。多一个少一个都说明映射错了。
4. **交界连续**：延续段第一循环的容量，必须接在原段最后一循环附近——落差小于
   延续段自身衰减的 1/4。若配到一颗已经到寿命的电池，落差会是衰减量的量级。

两个不能用的字段（踩过的坑，写下来免得再踩）
--------------------------------------------
``barcode``：140 颗电池全是同一个常数 3707764736，是占位值，不是身份。
``channel_id``：第 5 位是批次内行号（b1c0 是 47、b2c7 是 56），跨批次必然不同。

最初就是拿 channel_id 当身份，得到「5 对全 FAIL」。那不是延续关系不成立，
是字段选错了。

用法
----
    python scripts/check_continuations.py
    python scripts/check_continuations.py --source-dir data/mit-stanford/raw --workers 6

退出码：0 全部通过 / 1 有证据不成立 / 2 数据源出错
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flsoh.matr import MatrBatch, MatrError  # noqa: E402

from build_manifest import (  # noqa: E402
    CONTINUATIONS,
    CONTINUATION_ADD_LEN,
    CONTINUATION_LEN_SLACK,
    policy_candidates,
    policy_suffix_match,
    resolve_source,
)

# 交界落差的容差：不能超过延续段自身衰减的这个比例。
JUNCTION_RATIO = 0.25
# 绝对上限，防止衰减极小的段把比例判据放得太松。
JUNCTION_ABS_MAX = 0.05


def _read_policies(source: str, batch_no: int, indices: list[int]) -> list[dict]:
    """一个 worker 只读 policy，各开各的 h5py 句柄（h5py 不是线程安全的）。

    刻意只读这一条。``cell()`` 会连带读 channel_id / barcode / cycle_life /
    summary shape，而前三个**都不是身份**（barcode 是常数，channel_id 是批次内
    行号），读了也没用。要判断「后缀圈出的小组是否唯一」必须看批次 1 的全部
    46 颗，46 × 4 次无用往返 ≈ 3 分钟——远端读的成本全在往返上。
    """
    out = []
    with MatrBatch(source, batch_no=batch_no) as batch:
        for index in indices:
            out.append(
                {
                    "cell_id": f"b{batch_no}c{index}",
                    "policy": batch.policy(index),
                }
            )
    return out


def collect_policies(source: str, batch_no: int, indices: list[int], workers: int) -> dict:
    chunk = max(1, len(indices) // max(1, workers))
    chunks = [indices[i : i + chunk] for i in range(0, len(indices), chunk)]
    rows: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_read_policies, source, batch_no, c) for c in chunks]
        for future in futures:
            for row in future.result():
                rows[row["cell_id"]] = row
    return rows


def collect_n_cycles(source: str, batch_no: int, indices: list[int]) -> dict:
    """只要循环数，且只要那 5 颗——46 颗都读循环数是纯浪费。"""
    out = {}
    with MatrBatch(source, batch_no=batch_no) as batch:
        for index in indices:
            out[f"b{batch_no}c{index}"] = batch.n_cycles(index)
    return out


def read_edges(source: str, batch_no: int, indices: list[int]) -> dict:
    """首末循环的放电容量。只读 QDischarge，一次往返一颗。"""
    out = {}
    with MatrBatch(source, batch_no=batch_no) as batch:
        for index in indices:
            qd = batch.summary(index, fields=("QDischarge",))["QDischarge"]
            out[f"b{batch_no}c{index}"] = (float(qd[0]), float(qd[-1]))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="验证跨批次延续关系")
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)

    origin_ids = sorted({int(c.split("c")[1]) for c in CONTINUATIONS.values()})
    cont_ids = sorted({int(c.split("c")[1]) for c in CONTINUATIONS})
    source1 = resolve_source(1, args.source_dir)
    with MatrBatch(source1, batch_no=1) as probe:
        batch1_n = probe.n_cells

    print(f"[batch 1] {source1}  共 {batch1_n} 颗", flush=True)
    # 要判断「后缀圈出的小组是否唯一」，必须看批次 1 的**全部** policy。
    # 只读 5 颗的话，「唯一」这件事根本无从谈起。
    meta1 = collect_policies(source1, 1, list(range(batch1_n)), args.workers)
    print(f"  读到 {len(meta1)} 颗的 policy", flush=True)

    print(f"[batch 2] {resolve_source(2, args.source_dir)}", flush=True)
    meta2 = collect_policies(
        resolve_source(2, args.source_dir), 2, cont_ids, args.workers
    )
    cycles2 = collect_n_cycles(resolve_source(2, args.source_dir), 2, cont_ids)
    for cell_id in cont_ids:
        meta2[f"b2c{cell_id}"]["n_cycles"] = cycles2[f"b2c{cell_id}"]
    print(f"  读到 {len(meta2)} 颗的 policy 与循环数", flush=True)

    policies = {cid: row["policy"] for cid, row in meta1.items()}
    for cell_id in sorted(CONTINUATIONS):
        print(
            f"  {cell_id:<6s} n={meta2[cell_id]['n_cycles']:<5d} "
            f"policy={meta2[cell_id]['policy']!r}",
            flush=True,
        )

    print("\n[交界容量] 读 10 颗的首末 QDischarge", flush=True)
    edges = read_edges(resolve_source(1, args.source_dir), 1, origin_ids)
    edges.update(read_edges(resolve_source(2, args.source_dir), 2, cont_ids))

    problems: list[str] = []

    # —— 证据 1 + 2：身份与段长 ——
    print("\n证据 1/2：policy 后缀与段长指纹")
    print(f"  {'延续段':<7s}{'原段':<7s}{'policy 后缀':<10s}{'循环数':<8s}{'add_len+1':<10s}")
    for continuation, origin in CONTINUATIONS.items():
        left, right = meta2[continuation], meta1[origin]
        suffix_ok = policy_suffix_match(left["policy"], right["policy"])
        expected = CONTINUATION_ADD_LEN[continuation] + CONTINUATION_LEN_SLACK
        len_ok = left["n_cycles"] == expected
        print(
            f"  {continuation:<7s}{origin:<7s}"
            f"{'OK' if suffix_ok else 'FAIL':<10s}"
            f"{left['n_cycles']:<8d}{expected:<10d}"
            f"{'OK' if len_ok else 'FAIL'}"
        )
        if not suffix_ok:
            problems.append(
                f"{continuation} 的 policy {left['policy']!r} 不是 "
                f"{origin} 的 {right['policy']!r} 的后缀"
            )
        if not len_ok:
            problems.append(
                f"{continuation} 有 {left['n_cycles']} 个循环，按 add_len 应为 {expected}"
            )

    # —— 证据 3：组内 1:1 ——
    print("\n证据 3：同一条 policy 后缀圈出的原段必须正好对上延续段")
    groups: dict[str, list[str]] = {}
    for continuation in CONTINUATIONS:
        groups.setdefault(meta2[continuation]["policy"], []).append(continuation)
    for policy, members in sorted(groups.items()):
        candidates = policy_candidates(policy, policies)
        origins = sorted(CONTINUATIONS[m] for m in members)
        ok = candidates == origins
        print(
            f"  {policy!r:<14s} 圈出 {candidates} vs 应为 {origins}  "
            f"{'OK' if ok else 'FAIL'}"
        )
        if not ok:
            problems.append(
                f"policy {policy!r} 圈出 {candidates}，但延续段指向 {origins}"
            )

    # —— 证据 4：交界连续 ——
    print("\n证据 4：交界处容量落差 vs 延续段自身衰减")
    for continuation, origin in CONTINUATIONS.items():
        o_start, o_end = edges[origin]
        c_start, c_end = edges[continuation]
        gap = abs(o_end - c_start)
        fade = max(1e-9, c_start - c_end)
        ok = gap <= JUNCTION_ABS_MAX and gap <= JUNCTION_RATIO * fade
        print(
            f"  {origin} 末 {o_end:.4f} → {continuation} 首 {c_start:.4f}  "
            f"落差 {gap:.4f}  该段衰减 {fade:.4f}  比 {gap / fade:.2f}  "
            f"{'OK' if ok else 'FAIL'}"
        )
        if not ok:
            problems.append(
                f"{origin}→{continuation} 交界落差 {gap:.4f} 太大"
                f"（该段衰减 {fade:.4f}），接不上"
            )

    print()
    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        print(f"\n{len(problems)} 项证据不成立。不要按 CONTINUATIONS 删除任何电池。")
        return 1
    print(
        f"{len(CONTINUATIONS)} 对全部通过四条证据：这 5 颗确实是同一颗电池的另一段。\n"
        "继续按 build_manifest.py 的默认行为去重（保留批次 1 的那一段）。"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MatrError as exc:
        print(f"失败：{exc}", file=sys.stderr)
        raise SystemExit(2)
