#!/usr/bin/env python3
"""生成模拟 App 会话（Schema v1 JSONL）与真值。

    python3 scripts/simulate_sessions.py --cells 8 --cycles 20 \\
        --out data/simulated/sessions.jsonl --truth data/simulated/truth.json

产出两件东西：

1. ``--out``：一行一个会话的 JSONL，直接喂给 ``run_e2e.py`` 或
   ``check_schema.py``。**入库前必须先过 schema 校验**——模拟器自己造的数据
   也照样要过闸门，否则「契约」只约束别人。
2. ``--truth``：每个会话的真值容量与 SOH。没有它，端到端跑出来的 MAE 无法
   判断是「管道接对了」还是「模型碰巧输出了一个接近 1 的数」。

``--validate``（默认开）会把生成的会话过一遍 :func:`flsoh.schema.validate_sessions`，
有任何一条不合法就以退出码 1 结束，并把前几条错误打出来。**不要关掉它**：
模拟器一旦产出违反契约的数据，联调的所有结论都建立在错误的前提上，而
``run_e2e.py`` 不会替你再查一遍。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flsoh.schema import validate_sessions  # noqa: E402
from flsoh.simulate import (  # noqa: E402
    simulate_fleet,
    write_sessions_jsonl,
    write_truth_json,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="生成模拟 App 会话（Schema v1）与真值",
    )
    parser.add_argument("--cells", type=int, default=8, help="电池颗数（默认 8）")
    parser.add_argument("--cycles", type=int, default=20, help="每颗循环数（默认 20）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    parser.add_argument(
        "--dt-s", type=float, default=10.0, help="标称采样间隔秒（默认 10，带抖动）"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/simulated/sessions.jsonl"),
        help="会话 JSONL 输出路径",
    )
    parser.add_argument(
        "--truth", type=Path, default=Path("data/simulated/truth.json"),
        help="真值 JSON 输出路径",
    )
    parser.add_argument(
        "--no-validate", action="store_true",
        help="跳过 Schema 校验（不建议：模拟数据同样要过契约闸门）",
    )
    args = parser.parse_args(argv)

    sessions, truths = simulate_fleet(
        args.cells, n_cycles=args.cycles, seed=args.seed, dt_s=args.dt_s
    )
    if not args.no_validate:
        report = validate_sessions(sessions)
        if report["invalid"]:
            print(
                f"[simulate] FAIL：{report['invalid']}/{report['total']} 份会话违反 "
                "Schema v1。模拟器造的数据也要过闸门——不过的话，联调的所有结论"
                "都建立在错误前提上。",
                file=sys.stderr,
            )
            for item in report["sample_errors"][:5]:
                print(f"    #{item['index']}: {item['errors'][0]}", file=sys.stderr)
            return 1
        print(f"[simulate] Schema v1 校验：{report['valid']}/{report['total']} 份合法")

    write_sessions_jsonl(sessions, args.out)
    write_truth_json(truths, args.truth)
    n_records = sum(len(s["records"]) for s in sessions)
    soh = [t.get("soh") for t in truths]
    if any(value is None for value in soh):
        # 真值缺 soh 说明生成器与写盘契约脱节。照样把文件写出去（调用方可能只要
        # 会话），但必须报出来并给非零退出码——静默少一个字段比报错更贵。
        print(
            "[simulate] 真值里缺 soh 字段：会话已写出，但这份真值无法用于评估",
            file=sys.stderr,
        )
        return 2
    print(
        f"[simulate] {args.cells} 颗 × {args.cycles} 循环 = {len(sessions)} 份会话，"
        f"{n_records} 条记录"
    )
    print(
        f"[simulate] SOH 范围 [{min(soh):.5f}, {max(soh):.5f}]；"
        f"写入 {args.out} 与 {args.truth}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
