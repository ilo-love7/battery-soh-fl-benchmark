#!/usr/bin/env python3
"""校验一份 JSONL 会话文件是否符合 Schema v1（plan.md §8.1）。

    python scripts/check_schema.py sessions.jsonl
    python scripts/check_schema.py sessions.jsonl --verbose   # 打印每条错误

文件格式：一行一个 JSON object（JSONL）。空行与 ``#`` 注释行跳过。

退出码：0 = 全部合法；1 = 有非法会话；2 = 文件读不了 / 不是 JSONL。

**为什么单独一个脚本而不是库函数**：入库是 App 网关侧的动作，那边可能只有
标准库。`flsoh.schema` 纯标准库，这个脚本也是——检查契约的代码不该比被检查的
数据更依赖环境。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flsoh.schema import SCHEMA_VERSION, validate_sessions  # noqa: E402


def iter_sessions(path: Path):
    """逐行 yield；坏行直接抛出，不跳过。

    跳过坏行会让「文件少了 30%」这件事在计数上完全看不出来——与
    ``build_features.py`` 不静默跳过丢弃曲线是同一条理由。
    """
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {lineno} 行不是合法 JSON：{exc}") from exc
            yield lineno, obj


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="校验充电会话 JSONL（Schema v1）")
    ap.add_argument("sessions", type=Path, help="JSONL 文件，一行一个会话")
    ap.add_argument("--verbose", action="store_true", help="打印每条错误")
    args = ap.parse_args(argv)

    if not args.sessions.is_file():
        print(f"FAIL 文件不存在：{args.sessions}")
        return 2

    try:
        report = validate_sessions(obj for _, obj in iter_sessions(args.sessions))
    except ValueError as exc:
        print(f"FAIL {exc}")
        return 2

    print(f"Schema {SCHEMA_VERSION}  {args.sessions}")
    print(f"  共 {report['total']} 个会话：合法 {report['valid']}，"
          f"非法 {report['invalid']}")
    if report["sample_errors"]:
        print(f"  前 {len(report['sample_errors'])} 条错误：")
        for item in report["sample_errors"]:
            print(f"    #{item['index']}: {item['errors'][0]}")
            if args.verbose:
                for e in item["errors"][1:]:
                    print(f"        {e}")

    if report["invalid"]:
        print("FAIL 有会话不符合 Schema v1；入库前必须修数据或修采集端")
        return 1
    print("PASS 全部会话符合 Schema v1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
