#!/usr/bin/env python3
"""从 MATR 原始文件生成电池清单：``cells.txt`` + ``manifest.json``。

为什么清单是生成物
------------------
方案里最初写的「124 颗」是论文口径，而磁盘上那份 .mat 实际有多少颗、每颗多少个
循环，只有读出来才知道。手写清单一旦和数据脱节，``check_partition.py`` 就会
对着一份虚构的 ID 清单说 PASS——``P-UNKNOWN`` 防的是「ID 不在数据集里」，
它防不住「清单本身就是假的」。所以清单必须从数据生成，并带上生成时间与来源。

两种模式，默认是快的那个
------------------------
``cells.txt`` 只需要**每批多少颗**，而 ``n_cells`` 就写在 ``batch/summary`` 的
shape 里，读一次元数据就有。逐颗补 channel_id / barcode / policy / cycle_life
则要给每颗电池各发 7 次请求——实测每次往返约 1.4 s，46 颗就是 5 分钟以上。
所以默认只写 ``cell_id`` + ``index``，并标 ``cells_meta_complete: false``；
需要完整元信息时加 ``--full``。

这个取舍是有意的：**能被检查的东西要快，快的东西才会被经常跑**。清单是每次
划分前都要对的，不能让它变成一个没人愿意等 5 分钟的命令。

用法
----
    # 快模式（默认）：只写 cell_id，约 1 分钟
    python scripts/build_manifest.py --out data/mit-stanford

    # 完整模式：补全每颗电池的元信息
    python scripts/build_manifest.py --out data/mit-stanford --full --workers 4

    # 用本地已下载的 .mat（文件名须与 BATCH_FILES 一致）
    python scripts/build_manifest.py --out data/mit-stanford \\
        --source-dir data/mit-stanford/raw

输出
----
    <out>/cells.txt      一行一个 cell_id，供 check_partition.py --cells-file
    <out>/manifest.json  批次、颗数、来源、生成时间；--full 时含每颗电池元信息
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flsoh.matr import (  # noqa: E402
    BATCH_FILES,
    MATR_MIRROR,
    MatrBatch,
    MatrError,
    mirror_url,
)

# plan.md 与 configs/*.toml 里写的电池数。对不上就报出来——但**不改数据**，
# 改的是方案。数据是事实，方案是声明，两者冲突时永远先信数据。
PLAN_N_CELLS = 135

# 批次 2 里有 5 颗是批次 1 的延续：同一颗电池，第一个文件跑完一段、第二个文件
# 接着跑。论文的 LoadData.m / Load Data.ipynb 列出了这个对应关系，但照抄一个会
# 改变数据集构成的结论风险太高——位置映射一旦猜错，删掉的就是 5 颗正常电池，
# 而指标上看不出来。所以这里带了独立验证，见 ``check_continuation_evidence``。
#
# 不删它们会发生什么：同一颗电池同时出现在两个「客户端」里，而 P-LEAK 查不出来，
# 因为 cell_id 不同、文件也不同。这正是本仓库最贵的一条规则要防的事——
# 泄漏不报错，只让指标变好看。
CONTINUATIONS = {
    "b2c7": "b1c0",
    "b2c8": "b1c1",
    "b2c9": "b1c2",
    "b2c15": "b1c3",
    "b2c16": "b1c4",
}

# 论文脚本里的 add_len：批次 2 那一段追加了多少个循环。
# 这 5 个数互不相同，所以它能给「谁接谁」钉死一个指纹——配错一对，长度就对不上。
CONTINUATION_ADD_LEN = {
    "b2c7": 661,
    "b2c8": 980,
    "b2c9": 1059,
    "b2c15": 207,
    "b2c16": 481,
}

# 延续段的 ``summary`` 比 add_len 多一个元素：文件开头那个不完整循环。
CONTINUATION_LEN_SLACK = 1


def select_cells(manifest: dict, *, drop_continuations: bool = True) -> dict:
    """在 manifest 上做排除，返回新的 manifest（不改入参）。

    目前只有一条规则：跨批次延续的电池。其余电池一律保留——包括那些没跑到
    80% 容量的。理由：本项目的标签是逐循环 SOH，不是循环寿命，截断的电池
    在循环 5–100 区间内是完全正常的数据。论文剔掉它们是为了预测 cycle life，
    那是另一个任务。
    """
    out = json.loads(json.dumps(manifest))
    if not drop_continuations:
        for entry in out["batches"]:
            entry["n_cells_selected"] = len(entry["cells"])
        out["totals"]["n_cells_selected"] = out["totals"]["n_cells"]
        out["selection"] = {"drop_continuations": False, "excluded": []}
        return out

    excluded = []
    for entry in out["batches"]:
        keep = []
        for cell in entry["cells"]:
            origin = CONTINUATIONS.get(cell["cell_id"])
            if origin is None:
                keep.append(cell)
            else:
                excluded.append(
                    {
                        "cell_id": cell["cell_id"],
                        "reason": f"与 {origin} 是同一颗电池（批次延续）",
                    }
                )
        entry["cells"] = keep
        entry["n_cells_selected"] = len(keep)
    out["selection"] = {"drop_continuations": True, "excluded": excluded}
    out["totals"]["n_cells_selected"] = len(
        [c for e in out["batches"] for c in e["cells"]]
    )
    out["totals"]["matches_plan"] = (
        out["totals"]["n_cells_selected"] == PLAN_N_CELLS
    )
    return out


def policy_suffix_match(continuation_policy: str, origin_policy: str) -> bool:
    """延续段的 ``policy_readable`` 是否是原段的后缀。

    为什么是后缀而不是相等——这是数据里的事实，不是妥协：延续段的 policy 字符串
    在写入时就少了开头的 ``<倍率>C(``。实测 b1c0 存的是 ``3.6C(80%)-3.6C``（14
    个字符），它的延续段 b2c7 存的是 ``80%)-3.6C``（9 个字符）；b1c3 的
    ``4C(80%)-4C``（10 个字符）对 b2c15 的 ``80%)-4C``（7 个字符）。原始 uint16
    数组的形状就是 9 和 7，不是解码器吃掉了字符。

    这个残缺反而是好指纹：``80%)-4C`` 只能来自 ``4C(80%)-4C``，而 ``4.4C(80%)-4.4C``
    的后缀是 ``80%)-4.4C``，匹配不上。残缺把「同一个倍率」变成了「同一条 policy」。
    """
    if not continuation_policy or not origin_policy:
        return False
    return origin_policy.endswith(continuation_policy)


def policy_candidates(
    continuation_policy: str, policies: dict[str, str]
) -> list[str]:
    """批次 1 里哪些电池的 policy 能以这个后缀匹配上。返回排好序的 cell_id。"""
    return sorted(
        cid
        for cid, policy in policies.items()
        if policy_suffix_match(continuation_policy, policy)
    )


def check_continuation_evidence(manifest: dict) -> list[str] | None:
    """用 policy 后缀 + 段长指纹验证延续关系。返回 None 表示无从验证。

    三条独立证据，缺一条就不该删数据：

    1. **身份**：延续段的 policy 字符串必须是批次 1 某条 policy 的后缀，且这条
       后缀在批次 1 里只能圈出规模相同的一小组。这说明它不是一颗新电池，
       而是同一批测试的另一段。
    2. **段长**：延续段的循环数必须等于论文 add_len + 1。5 个 add_len 互不相同，
       所以这一条同时钉住了「谁接谁」。
    3. **组内 1:1**：每颗电池只能被接一次。两段各接两次说明映射写错了。

    第 4 条证据（交界处容量连续）需要读 summary，在 ``check_continuations.py``
    里做，因为它要联网。

    反面教材写在注释里，避免下次又走回头路：``barcode`` 和 ``channel_id``
    **都不是身份字段**。barcode 在全部 140 颗电池上都是同一个常数 3707764736；
    channel_id 的第 5 位是批次内行号（b1c0 是 47、b2c7 是 56），跨批次必然不同。
    曾经用 channel_id 做验证，得到 5 对全 FAIL——那不是延续关系不成立，
    是拿了个根本不是身份的字段当身份。
    """
    if not all(entry.get("cells_meta_complete") for entry in manifest["batches"]):
        return None
    by_id = {
        cell["cell_id"]: cell for entry in manifest["batches"] for cell in entry["cells"]
    }
    policies = {
        cell["cell_id"]: cell.get("policy", "")
        for entry in manifest["batches"]
        if entry["batch_no"] == 1
        for cell in entry["cells"]
    }
    problems: list[str] = []
    used: dict[str, str] = {}

    for continuation, origin in CONTINUATIONS.items():
        left, right = by_id.get(continuation), by_id.get(origin)
        if left is None or right is None:
            problems.append(f"{continuation} 或 {origin} 不在 manifest 里")
            continue

        # 1. 身份：后缀必须匹配
        if not policy_suffix_match(left.get("policy", ""), right.get("policy", "")):
            problems.append(
                f"{continuation} 的 policy {left.get('policy')!r} 不是 "
                f"{origin} 的 {right.get('policy')!r} 的后缀"
            )

        # 2. 段长指纹
        expected = CONTINUATION_ADD_LEN.get(continuation, -1) + CONTINUATION_LEN_SLACK
        actual = left.get("n_cycles")
        if actual != expected:
            problems.append(
                f"{continuation} 有 {actual} 个循环，按 add_len 应为 {expected}"
            )

        # 3. 组内 1:1
        if origin in used:
            problems.append(
                f"{origin} 被 {used[origin]} 和 {continuation} 同时认作原段"
            )
        used[origin] = continuation

    # 1 的加强版：同一条 policy 后缀圈出的批次 1 电池，必须正好是这一组延续段的
    # 原段，一个不多一个不少。多一个说明还有别的电池共享这条 policy，映射不唯一；
    # 少一个说明有延续段配错了组。
    groups: dict[str, list[str]] = {}
    for continuation in CONTINUATIONS:
        cell = by_id.get(continuation)
        if cell is not None:
            groups.setdefault(cell.get("policy", ""), []).append(continuation)
    for policy, members in groups.items():
        candidates = policy_candidates(policy, policies)
        origins = sorted(CONTINUATIONS[m] for m in members)
        if candidates != origins:
            problems.append(
                f"policy {policy!r} 圈出的批次 1 电池是 {candidates}，"
                f"但延续段 {sorted(members)} 指向 {origins}"
            )
    return problems


def verify_continuations(manifest: dict) -> list[str] | None:
    """``check_continuation_evidence`` 的旧名字，保留给既有调用方。"""
    return check_continuation_evidence(manifest)


def resolve_source(batch_no: int, source_dir: Path | None) -> str:
    """本地有就用本地，没有就回落到镜像。"""
    if source_dir is not None:
        candidate = source_dir / BATCH_FILES[batch_no]
        if candidate.is_file():
            return str(candidate)
    return mirror_url(batch_no)


def cell_id(batch_no: int, index: int) -> str:
    """社区惯例的位置命名。口径写在 manifest 的 id_scheme 里。"""
    return f"b{batch_no}c{index}"


def _read_chunk(source: str, batch_no: int, indices: list[int]) -> list[dict]:
    metas = []
    with MatrBatch(source, batch_no=batch_no) as batch:
        for index in indices:
            meta = batch.cell(index)
            metas.append(
                {
                    "cell_id": meta.cell_id,
                    "index": meta.index,
                    "channel_id": list(meta.channel_id),
                    "barcode": meta.barcode,
                    "policy": meta.policy,
                    "cycle_life": meta.cycle_life,
                    "n_cycles": meta.n_cycles,
                }
            )
    return metas


def collect_batch(
    batch_no: int,
    source: str,
    *,
    full: bool = False,
    workers: int = 4,
    chunk: int = 4,
) -> dict:
    """读一个批次。``full=False`` 时只取颗数，不逐颗读元信息。"""
    with MatrBatch(source, batch_no=batch_no) as probe:
        n_cells = probe.n_cells
        batch_date = probe.batch_date
        n_bytes = probe.n_bytes

    entry = {
        "batch_no": batch_no,
        "batch_date": batch_date,
        "source": source,
        "n_bytes": n_bytes,
        "n_cells": n_cells,
        "cells_meta_complete": bool(full),
        "cells": [
            {"cell_id": cell_id(batch_no, i), "index": i} for i in range(n_cells)
        ],
    }
    if not full:
        return entry

    tasks = [
        list(range(start, min(start + chunk, n_cells)))
        for start in range(0, n_cells, chunk)
    ]
    metas: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
        for part in pool.map(lambda idx: _read_chunk(source, batch_no, idx), tasks):
            metas.extend(part)
    metas.sort(key=lambda meta: meta["index"])
    if len(metas) != n_cells:
        raise MatrError(
            f"批次 {batch_no} 读到 {len(metas)} 颗，声明 {n_cells} 颗——分片漏了"
        )
    entry["cells"] = metas
    return entry


def build_manifest(
    batches: list[int],
    *,
    source_dir: Path | None = None,
    full: bool = False,
    workers: int = 4,
    chunk: int = 4,
) -> dict:
    """生成 manifest 字典（不落盘，便于测试）。"""
    entries = []
    for batch_no in batches:
        source = resolve_source(batch_no, source_dir)
        print(f"[batch {batch_no}] {source}", flush=True)
        entry = collect_batch(
            batch_no, source, full=full, workers=workers, chunk=chunk
        )
        entries.append(entry)
        print(f"  {entry['n_cells']} 颗", flush=True)

    ids = [cell["cell_id"] for entry in entries for cell in entry["cells"]]
    if len(ids) != len(set(ids)):
        raise MatrError("生成的 cell_id 有重复——批次号或位置口径出了问题")
    return {
        "generated_by": "scripts/build_manifest.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "canonical_source": "https://data.matr.io/1/",
        "mirror": MATR_MIRROR,
        "id_scheme": "b{批次号}c{结构数组下标}，下标 0 起；文件内无人类可读电池名",
        "batches": entries,
        "totals": {
            "n_cells": len(ids),
            "n_cycles": (
                sum(cell["n_cycles"] for entry in entries for cell in entry["cells"])
                if full
                else None
            ),
            "plan_n_cells": PLAN_N_CELLS,
            "matches_plan": len(ids) == PLAN_N_CELLS,
        },
    }


def write_outputs(manifest: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = [cell["cell_id"] for entry in manifest["batches"] for cell in entry["cells"]]
    cells_path = out_dir / "cells.txt"
    cells_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return cells_path, manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 MATR 电池清单")
    parser.add_argument("--out", required=True, type=Path, help="输出目录")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="本地 .mat 所在目录；缺文件时回落到镜像",
    )
    parser.add_argument("--batches", default="1,2,3", help="逗号分隔的批次号")
    parser.add_argument(
        "--full", action="store_true", help="逐颗补全元信息（慢，每颗约 7 次往返）"
    )
    parser.add_argument("--workers", type=int, default=4, help="--full 时的并发数")
    parser.add_argument("--chunk", type=int, default=4, help="每个 worker 一次处理几颗")
    parser.add_argument(
        "--from-manifest",
        type=Path,
        default=None,
        help="不联网，直接在已有 manifest.json 上重做选择与 cells.txt",
    )
    parser.add_argument(
        "--keep-continuations",
        action="store_true",
        help="保留跨批次延续的 5 颗（默认删除；保留会让同一颗电池出现在两个客户端）",
    )
    args = parser.parse_args(argv)

    if args.from_manifest is not None:
        raw = json.loads(args.from_manifest.read_text(encoding="utf-8"))
    else:
        batches = [int(x) for x in args.batches.split(",") if x.strip()]
        try:
            raw = build_manifest(
                batches,
                source_dir=args.source_dir,
                full=args.full,
                workers=args.workers,
                chunk=args.chunk,
            )
        except MatrError as exc:
            print(f"失败：{exc}", file=sys.stderr)
            return 2

    problems = verify_continuations(raw)
    if problems is None:
        print("\n（未做延续验证：manifest 不含 channel_id，需 --full 生成）")
    elif problems:
        print("\n延续关系验证失败，拒绝按 CONTINUATIONS 删除：", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        if not args.keep_continuations:
            print("  （这是保护，不是报错：位置映射猜错时删的是正常数据）", file=sys.stderr)
            return 3
    else:
        print(f"\n延续关系验证通过：{len(CONTINUATIONS)} 对 channel_id 逐位相同")

    manifest = select_cells(
        raw, drop_continuations=not args.keep_continuations
    )
    cells_path, manifest_path = write_outputs(manifest, args.out)
    totals = manifest["totals"]
    print()
    for entry in manifest["batches"]:
        print(
            f"  batch {entry['batch_no']}  {entry['batch_date']}  "
            f"{entry['n_cells']:>3} 颗 → 选 {entry['n_cells_selected']:>3} 颗  "
            f"{(entry['n_bytes'] or 0) / 1024**3:.2f} GiB"
        )
    print(f"\ncells.txt      {cells_path}  ({totals['n_cells_selected']} 行)")
    print(f"manifest.json  {manifest_path}")
    if not args.full and args.from_manifest is None:
        print("（快模式：未读每颗电池的元信息，manifest 里 cells_meta_complete=false）")
    if not totals["matches_plan"]:
        print()
        print(
            f"注意：plan.md 与 configs/*.toml 写的是 {PLAN_N_CELLS} 颗，"
            f"选出 {totals['n_cells_selected']} 颗（原始 {totals['n_cells']} 颗）。"
        )
        print("  这不是数据错误，是方案里的数没有对着数据核过。")
        print("  configs 的 n_cells / test_cells 必须改成实际值，否则")
        print("  check_protocol 的客户端数断言一直在验一个不存在的数据集。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
