#!/usr/bin/env python3
"""划分泄漏检测 CLI（docs/plan.md §6.3）。

两种用法，对应两个不同的问题：

1. **检查生成器**——按配置生成划分，再检查这份划分干不干净：

       python3 scripts/check_partition.py --config configs/mvp.toml

2. **检查产物**——训练实际用的那份划分文件，是不是干净：

       python3 scripts/check_partition.py --config configs/mvp.toml \
           --partition-json results/partition-mvp-k5-dirichlet0.5.json

   第 2 种才是真正的对账。第 1 种只能证明"脚本没问题"，不能证明"你跑的
   那次没问题"——中间可能有人手改过划分文件。

另外 ``--emit`` 把生成好的划分写成 JSON，供训练脚本直接读：

    python3 scripts/check_partition.py --config configs/mvp.toml \
        --emit results/partition-mvp-k5-iid.json --emit-k 5 --emit-scheme iid

规则 1「同一颗电池不得跨客户端」靠人眼核对 135 颗电池不现实，所以做成脚本。
这一步必须跑在训练之前：划分错了，后面所有实验都白做，而且从精度上看不出来
——只会看到一个"效果很好"的假结论。

退出码：0 = 全 PASS，1 = 有检查项失败，2 = 配置/文件读不出来或协议有 ERROR。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flsoh.partition import (  # noqa: E402
    PartitionError,
    check_partition,
    expected_n_clients,
    make_cell_ids,
    partition_by_soh_rank,
    partition_dirichlet,
    partition_iid,
    partition_sizes,
    split_test_cells,
)
from flsoh.protocol import (  # noqa: E402
    ProtocolError,
    has_errors,
    load_config,
    validate_config,
)


def _load_cell_soh(path: str | None) -> dict[str, float] | None:
    """读 `{cell_id: 平均 SOH}`。由 scripts/measure_heterogeneity.py 生成。"""
    if not path:
        return None
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PartitionError(f"读不出 {path}：{exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise PartitionError(f"{path} 必须是 {{cell_id: soh}} 的非空对象")
    return {str(k): float(v) for k, v in raw.items()}


def _load_cells(path: str | None, n_cells: int) -> list[str]:
    if path is None:
        return make_cell_ids(n_cells)
    raw = Path(path).read_text(encoding="utf-8").splitlines()
    cells = [line.strip() for line in raw if line.strip() and not line.startswith("#")]
    if not cells:
        raise PartitionError(f"{path} 里没有电池 ID")
    return cells


def _n_clients_for(cfg: dict, n_available: int, k: int) -> int:
    """客户端数由「可用电池数 ÷ k」决定，不另设一个声明值。

    另设声明值的话，声明与实际一旦分叉，P-COUNT 抓到的是"配置写错"而不是
    "划分错了"——两种错误混在一个码里，排查时会被带偏。
    """
    return expected_n_clients(
        n_available,
        0,
        k,
        str(cfg.get("partitions", {}).get("remainder", "spread")),
    )


def _build(cfg, cells, k, scheme, alpha, seed, cell_soh=None):
    n_clients = _n_clients_for(cfg, len(cells), k)
    if scheme == "iid":
        return partition_iid(cells, n_clients, seed), n_clients
    if scheme == "dirichlet":
        if alpha is None:
            raise PartitionError("dirichlet 方案缺少 alpha")
        return partition_dirichlet(cells, n_clients, alpha, seed), n_clients
    if scheme == "sohrank":
        if not cell_soh:
            raise PartitionError(
                "sohrank 方案需要 --cell-soh-json（每颗电池的平均 SOH）"
            )
        return partition_by_soh_rank(cells, n_clients, cell_soh, seed), n_clients
    raise PartitionError(f"未知划分方案 {scheme!r}")


def _load_partition_json(path: str) -> tuple[dict[str, list[str]], list[str]]:
    """读一份划分产物。兼容两种形状：

    ``{"partition": {...}, "test_cells": [...]}``（--emit 的输出）
    ``{"client-0": [...], ...}``（只有客户端部分）
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PartitionError(f"{path} 不是合法 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise PartitionError(f"{path} 顶层必须是对象")

    if "partition" in payload:
        partition = payload["partition"]
        test_cells = payload.get("test_cells", []) or []
    else:
        partition = {k: v for k, v in payload.items() if isinstance(v, list)}
        test_cells = payload.get("test_cells", []) or []

    if not isinstance(partition, dict) or not partition:
        raise PartitionError(f"{path} 里没有可识别的客户端划分")
    for cid, cells in partition.items():
        if not isinstance(cells, list):
            raise PartitionError(f"{path}: 客户端 {cid} 的值不是列表")
    return partition, list(test_cells)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检测客户端划分是否泄漏")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--cells-file", default=None, help="一行一个电池 ID；省略则用占位 ID")
    parser.add_argument("--partition-json", default=None, help="对账一份已有的划分产物")
    parser.add_argument("--emit", default=None, help="把生成的划分写成 JSON")
    parser.add_argument("--emit-k", type=int, default=None, help="--emit 用哪档 k，默认取第一档")
    parser.add_argument("--emit-scheme", default="iid", help="--emit 用哪个方案")
    parser.add_argument("--emit-alpha", type=float, default=None, help="--emit 的 Dirichlet α")
    parser.add_argument("--cell-soh-json", default=None,
                        help="每颗电池的平均 SOH（sohrank 方案需要）")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ProtocolError as exc:
        print(f"FAIL {args.config}: {exc}")
        return 2

    proto_issues = validate_config(cfg)
    if has_errors(proto_issues):
        print(f"FAIL {args.config}: 协议有 ERROR，先修协议再查划分")
        for level, code, message in proto_issues:
            if level == "ERROR":
                print(f"  [{level}] {code}: {message}")
        return 2

    n_cells = int(cfg["data"]["n_cells"])
    try:
        all_cells = _load_cells(args.cells_file, n_cells)
    except PartitionError as exc:
        print(f"FAIL 读取电池清单：{exc}")
        return 2

    if len(all_cells) != n_cells:
        print(
            f"WARN 电池清单有 {len(all_cells)} 个 ID，与 data.n_cells={n_cells} 不一致；"
            "以清单为准，但协议里的 N 需要同步"
        )

    parts = cfg["partitions"]
    split_seed = int(parts.get("split_seed", 0))
    try:
        client_cells, test_cells = split_test_cells(
            all_cells, int(parts.get("test_cells", 0)), split_seed
        )
    except PartitionError as exc:
        print(f"FAIL 留出测试集：{exc}")
        return 2

    # ---- 模式 2：对账已有产物 ----
    if args.partition_json:
        try:
            partition, declared_test = _load_partition_json(args.partition_json)
        except PartitionError as exc:
            print(f"FAIL 读取划分产物：{exc}")
            return 2
        # 产物自带的 test_cells 优先：它才是那次训练实际用的测试集
        used_test = declared_test or test_cells
        issues = check_partition(
            partition,
            all_cells=all_cells,
            test_cells=used_test,
            expected_clients=None,
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "config": args.config,
                        "partition_json": args.partition_json,
                        "clients": len(partition),
                        "test_cells": len(used_test),
                        "status": "PASS" if not issues else "FAIL",
                        "issues": [{"code": c, "message": m} for c, m in issues],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 1 if issues else 0
        if issues:
            print(f"FAIL {args.partition_json}  N={len(partition)} test={len(used_test)}")
            for code, message in issues:
                print(f"  {code}: {message}")
            return 1
        print(f"PASS {args.partition_json}  N={len(partition)} test={len(used_test)}")
        return 0

    # ---- 模式 0：单独生成一份不在配置矩阵里的划分 ----
    # sohrank 是压力测试用的划分（它用到了标签），不该出现在
    # configs/mvp.toml 的 schemes 矩阵里——那会让每次普通检查都要带上它。
    # 但产物仍然必须过同一套泄漏检测，所以走这里而不是绕过去。
    if args.emit and args.emit_scheme not in parts.get("schemes", []):
        k = args.emit_k if args.emit_k is not None else parts["cells_per_client"][0]
        try:
            partition, n_clients = _build(
                cfg, client_cells, k, args.emit_scheme, args.emit_alpha,
                split_seed, _load_cell_soh(args.cell_soh_json),
            )
        except PartitionError as exc:
            print(f"FAIL 构造划分：{exc}")
            return 2
        issues = check_partition(
            partition,
            all_cells=all_cells,
            test_cells=test_cells,
            expected_clients=n_clients,
        )
        if issues:
            print(f"FAIL {args.emit_scheme} k={k} 划分没通过泄漏检测")
            for code, message in issues:
                print(f"  {code}: {message}")
            return 1
        out = Path(args.emit)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "config": args.config,
                    "k": k,
                    "scheme": args.emit_scheme,
                    "alpha": args.emit_alpha,
                    "split_seed": split_seed,
                    "test_cells": test_cells,
                    "partition": partition,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"已写出划分：{out}  N={len(partition)}  "
              f"（{args.emit_scheme} k={k}，泄漏检测 PASS）")
        return 0

    # ---- 模式 1：检查生成器 ----
    rows = []
    failed = 0
    emitted = None
    for k in parts.get("cells_per_client", []):
        for scheme in parts.get("schemes", []):
            alphas = parts.get("dirichlet_alpha", []) if scheme == "dirichlet" else [None]
            for alpha in alphas:
                try:
                    partition, n_clients = _build(
                        cfg, client_cells, k, scheme, alpha, split_seed
                    )
                except PartitionError as exc:
                    failed += 1
                    rows.append(
                        {
                            "k": k,
                            "scheme": scheme,
                            "alpha": alpha,
                            "status": "FAIL",
                            "issues": [{"code": "P-BUILD", "message": str(exc)}],
                        }
                    )
                    continue

                issues = check_partition(
                    partition,
                    all_cells=all_cells,
                    test_cells=test_cells,
                    expected_clients=n_clients,
                )
                if issues:
                    failed += 1
                sizes = partition_sizes(partition)
                rows.append(
                    {
                        "k": k,
                        "scheme": scheme,
                        "alpha": alpha,
                        "clients": len(partition),
                        "test_cells": len(test_cells),
                        "size_histogram": {str(s): n for s, n in sorted(sizes.items())},
                        "status": "PASS" if not issues else "FAIL",
                        "issues": [{"code": c, "message": m} for c, m in issues],
                    }
                )
                if args.emit and emitted is None:
                    want_k = args.emit_k if args.emit_k is not None else k
                    if k == want_k and scheme == args.emit_scheme:
                        if args.emit_alpha is None or alpha == args.emit_alpha:
                            emitted = {
                                "config": args.config,
                                "k": k,
                                "scheme": scheme,
                                "alpha": alpha,
                                "split_seed": split_seed,
                                "test_cells": test_cells,
                                "partition": partition,
                            }

    if args.emit:
        if emitted is None:
            print(f"FAIL --emit 没匹配到配置里的任何组合（k={args.emit_k} scheme={args.emit_scheme}）")
            return 2
        out = Path(args.emit)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(emitted, ensure_ascii=False, indent=2), encoding="utf-8")
        if not args.json:
            print(f"已写出划分：{out}  N={len(emitted['partition'])}")

    if args.json:
        print(
            json.dumps(
                {
                    "config": args.config,
                    "n_cells": len(all_cells),
                    "test_cells": len(test_cells),
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1 if failed else 0

    print(f"{Path(args.config).name}  cells={len(all_cells)}  test={len(test_cells)}")
    for row in rows:
        label = f"k={row['k']} {row['scheme']}"
        if row.get("alpha") is not None:
            label += f" alpha={row['alpha']}"
        if row["status"] == "PASS":
            hist = " ".join(f"{s}颗×{n}" for s, n in row["size_histogram"].items())
            print(f"  {row['status']}  {label:<26} N={row['clients']:<4} {hist}")
        else:
            print(f"  {row['status']}  {label:<26} N={row.get('clients', '?')}")
            for issue in row["issues"]:
                print(f"        {issue['code']}: {issue['message']}")

    if failed:
        print(f"\n{failed} 个配置未通过。划分错了，训练结果不能用——先修这里。")
        return 1
    print("\n全 PASS：无跨客户端泄漏、无测试集污染、无空客户端、无未覆盖电池。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
