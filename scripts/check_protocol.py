#!/usr/bin/env python3
"""实验矩阵校验 CLI（docs/plan.md §9.2）。

    python3 scripts/check_protocol.py --config configs/mvp.toml

退出码：0 = 无 ERROR（可能有 WARN），1 = 有 ERROR，2 = 配置读不出来。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flsoh.protocol import (  # noqa: E402
    LEVEL_ERROR,
    ProtocolError,
    expected_clients_for,
    has_errors,
    load_config,
    validate_config,
)


def _fmt(level: str, code: str, message: str) -> str:
    return f"  [{level}] {code}: {message}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验实验协议 TOML")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ProtocolError as exc:
        if args.json:
            print(json.dumps({"config": args.config, "error": str(exc)}, ensure_ascii=False))
        else:
            print(f"FAIL {args.config}: {exc}")
        return 2

    issues = validate_config(cfg)
    name = cfg.get("meta", {}).get("name", Path(args.config).stem)
    n_errors = sum(1 for level, _, _ in issues if level == LEVEL_ERROR)
    n_warns = len(issues) - n_errors

    # 顺带报出每档 k 的客户端数：这是报告里最容易写错的数字
    matrix = []
    for k in cfg.get("partitions", {}).get("cells_per_client", []) or []:
        if isinstance(k, int) and k > 0:
            matrix.append({"cells_per_client": k, "n_clients": expected_clients_for(cfg, k)})

    if args.json:
        print(
            json.dumps(
                {
                    "config": args.config,
                    "name": name,
                    "errors": n_errors,
                    "warnings": n_warns,
                    "issues": [
                        {"level": level, "code": code, "message": message}
                        for level, code, message in issues
                    ],
                    "clients": matrix,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1 if n_errors else 0

    status = "FAIL" if n_errors else "PASS"
    print(f"{status} {args.config}  ({name})  errors={n_errors} warnings={n_warns}")
    for level, code, message in issues:
        print(_fmt(level, code, message))
    if matrix:
        summary = "  ".join(f"k={m['cells_per_client']}→N={m['n_clients']}" for m in matrix)
        print(f"  客户端数：{summary}")
    if not issues:
        print("  协议齐备：划分原子、iid 对照、centralized 基线、seed 数、判据阈值均在场")
    return 1 if n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
