#!/usr/bin/env python3
"""把一次重跑的产物提升为仓库的正式结果目录，并拒绝「提升一半」。

    python3 scripts/promote_results.py \\
        --from results/discharge-v2 --to results \\
        --expect-feature-version discharge-v2 --threads 1 --execute

## 为什么需要一个脚本，而不是 `cp *.json`

因为 `cp` 有两种半途而废的方式，而两种都**看不出来**：

1. **少拷了几个**。重跑里有几个任务失败，`cp` 照样把其余 165 个拷过去，
   旧的 7 个留在原地。于是 `results/` 里 165 个新结果 + 7 个旧结果。
2. **口径不对**。重跑目录里混进了上一版的产物（`--skip-existing` 续跑时
   最可能），拷过去之后新旧在同一个目录里，谁也看不出来。

这两种情况的后果是同一条：`summarize_results.py --write-plan docs/plan.md`
会把**混杂的表**写进文档，而 CI 的 `--check-plan` 会**通过**——它只核对
「文档与 results/ 是否一致」，不核对「results/ 内部是否同源」。一张永久
写错的头版表，闸门全绿。

所以这里的每一道检查都是针对「提升了一半」这个失败模式：

| 检查 | 拦的是什么 |
|---|---|
| 源目录里每个 run 的 `feature_version` 等于期望值 | 源里混进了上一版产物 |
| 源目录里每个 run 的 `torch_num_threads` 等于期望值 | 线程数没钉住 |
| 目标目录里已有的 run **全部**出现在源里 | 少拷 → 新旧混杂 |
| 提升后重扫目标目录，不允许出现口径不符的 run | 上面三条漏掉的情况 |

前三条在 `--execute` 之前跑，任一条不过就退出 2、**一个文件都不拷**。
第四条在拷完之后跑，它是最后一道兜底：前三条都是「我以为我检查了」，
第四条是「拷完之后事实是什么」。

## 它不改什么

`partition-*.json`、`heterogeneity.json` 这类**划分产物**与特征口径无关，
原地不动。判据是「有没有 `final_mae`」——划分文件没有这个键，run 产物有。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

#: 只有带这个键的文件才算「一次实验的产物」。划分文件、heterogeneity 都没有。
MAE_KEY = "final_mae"


def load_runs(directory: Path) -> dict[str, dict]:
    """读出目录里所有实验产物。坏 JSON 与无 `final_mae` 的文件跳过。"""
    runs: dict[str, dict] = {}
    if not directory.is_dir():
        return runs
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and MAE_KEY in payload:
            runs[path.stem] = payload
    return runs


def check_source(
    runs: dict[str, dict],
    *,
    expect_feature_version: str | None,
    threads: int | None,
) -> list[str]:
    """源目录里每个 run 的口径都必须与期望值一致。

    期望值给 `None` 时不检查——读不出的期望值不能拿来下结论，这条规则和
    ``rerun_all.artifact_is_current`` 一样。但**要打印出来**，否则「没检查」
    会被读成「检查过了」。
    """
    problems: list[str] = []
    if expect_feature_version is not None:
        bad = sorted(
            rid for rid, r in runs.items()
            if r.get("feature_version") != expect_feature_version
        )
        if bad:
            problems.append(
                f"{len(bad)}/{len(runs)} 个 run 的 feature_version 不是 "
                f"{expect_feature_version!r}（例：{bad[:3]}）——源目录里混进了"
                "别的口径的产物，提升上去就是新旧混杂"
            )
    if threads is not None:
        bad = sorted(
            rid for rid, r in runs.items()
            if r.get("torch_num_threads") != threads
        )
        if bad:
            problems.append(
                f"{len(bad)}/{len(runs)} 个 run 的 torch_num_threads 不是 {threads}"
                f"（例：{bad[:3]}）——线程数没钉住，实测同一 run 在 1 线程与 "
                "16 线程下 final_mae 能差 1.8 倍"
            )
    return problems


def check_coverage(source: dict[str, dict], target: dict[str, dict]) -> list[str]:
    """目标目录里已有的 run 必须全部出现在源里。

    少一个就意味着「提升之后那个位置还是旧结果」，而目录看起来是完整的
    （文件数没变、每个 JSON 都合法、seeds 也齐）。这是最隐蔽的一种混杂：
    出错的是**集合**，不是数值。
    """
    missing = sorted(set(target) - set(source))
    if not missing:
        return []
    return [
        f"目标目录里有 {len(missing)} 个 run 在源目录里找不到（例：{missing[:3]}）——"
        "提升之后这些位置仍是旧结果，目录会变成新旧混杂，而文件数与 seeds 都正常。"
        "先把这些 run 补跑到源目录，或者确认它们已被有意废弃"
    ]


def check_after(target: dict[str, dict], *, expect_feature_version: str | None) -> list[str]:
    """拷完之后重扫目标目录：不允许出现口径不符的 run。

    前三道检查都是「按我的理解核对源目录」，这一道是「拷完之后事实上是什么」。
    两者都要有，因为前三道可能漏掉某个我没预料到的来源。
    """
    if expect_feature_version is None:
        return []
    bad = sorted(
        rid for rid, r in target.items()
        if r.get("feature_version") != expect_feature_version
    )
    if not bad:
        return []
    return [
        f"提升后目标目录里仍有 {len(bad)} 个 run 的 feature_version 不是 "
        f"{expect_feature_version!r}（例：{bad[:3]}）"
    ]


def promote(source_dir: Path, target_dir: Path, run_ids: list[str]) -> int:
    """把源目录里的这些 run 拷进目标目录，返回拷贝个数。

    只拷 `*.json`，且只拷 `run_ids` 里点过名的——不 `copytree`，因为源目录
    可能还带着日志、临时文件、上一版的目录结构。
    """
    copied = 0
    for rid in run_ids:
        src = source_dir / f"{rid}.json"
        if not src.is_file():
            continue
        shutil.copy2(src, target_dir / f"{rid}.json")
        copied += 1
    return copied


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="把重跑产物提升为正式结果目录，并拒绝「提升一半」",
    )
    ap.add_argument("--from", dest="source", required=True, help="重跑产物目录")
    ap.add_argument("--to", dest="target", required=True, help="正式结果目录")
    ap.add_argument("--expect-feature-version", default=None,
                    help="源目录里每个 run 都必须记着这个 feature_version")
    ap.add_argument("--threads", type=int, default=None,
                    help="源目录里每个 run 都必须记着这个 torch_num_threads")
    ap.add_argument("--execute", action="store_true", help="真拷；不给只做检查")
    args = ap.parse_args(argv)

    source_dir = Path(args.source)
    target_dir = Path(args.target)
    source = load_runs(source_dir)
    target = load_runs(target_dir)
    if not source:
        print(f"FAIL {source_dir} 里没有带 {MAE_KEY} 的产物")
        return 2
    print(f"源 {source_dir}：{len(source)} 个 run")
    print(f"目标 {target_dir}：{len(target)} 个 run（将被同名覆盖）")
    if args.expect_feature_version is None:
        print("WARN 没给 --expect-feature-version：不检查特征口径")
    if args.threads is None:
        print("WARN 没给 --threads：不检查线程数")

    problems = check_source(
        source,
        expect_feature_version=args.expect_feature_version,
        threads=args.threads,
    )
    problems += check_coverage(source, target)
    if problems:
        print("FAIL 提升前检查未通过（一个文件都没拷）：")
        for p in problems:
            print(f"  - {p}")
        return 2

    # check_coverage 已经保证 target ⊆ source，所以这里就是全部源 run。
    to_copy = sorted(source)
    if not args.execute:
        print(f"将拷贝 {len(to_copy)} 个 run 到 {target_dir}"
              "（未加 --execute，以上只是检查）")
        return 0

    target_dir.mkdir(parents=True, exist_ok=True)
    copied = promote(source_dir, target_dir, to_copy)
    print(f"已拷贝 {copied} 个 run")

    after = load_runs(target_dir)
    problems = check_after(after, expect_feature_version=args.expect_feature_version)
    if problems:
        print("FAIL 提升后复核未通过：")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"OK 提升完成：{target_dir} 里 {len(after)} 个 run，"
          f"口径全部为 {args.expect_feature_version!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
