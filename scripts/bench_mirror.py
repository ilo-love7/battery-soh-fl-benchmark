"""量远端镜像的吞吐与单颗耗时。**换时段跑一遍再引用任何数字。**

存在的理由：这个仓库的 docstring 里有一堆吞吐数字（1.5 次/秒、13 KB/s、
24 s/颗 vs 120 s/颗），而它们全是**时段相关**的。同一个镜像早上和晚上
能差 5 倍，把某一次的量测当成常量是这里犯过最贵的错。所以留一个能
复跑的工具，让下一个读 docstring 的人先量一遍再信。

三个子命令：

``rate``   纯 HTTP：给定并发下的请求速率与字节吞吐
``cell``   真实代码路径：抽 N 颗电池的墙钟时间（可指定顶层 workers）
``addrs``  单颗电池 134 个请求的地址分布，判断预读窗口是否成立

用法::

    python scripts/bench_mirror.py rate
    python scripts/bench_mirror.py cell --cells 2 --workers 1
    python scripts/bench_mirror.py addrs
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

HOST = "media.githubusercontent.com"
PATH = (
    "/media/DESN-Group-F/bsc-f2a5cc56/main/data/raw/original_sources/"
    "collection/raw/science/SRC-017/"
    "2017-05-12_batchdata_updated_struct_errorcorrect.mat"
)


# ---------------------------------------------------------------------------
# rate：纯 HTTP 层
# ---------------------------------------------------------------------------
_TLS = threading.local()


def _fetch_into(offset: int, size: int) -> int:
    """线程本地连接上抓一段。复用连接——这正是要验证的变量。"""
    conn = getattr(_TLS, "conn", None)
    if conn is None:
        conn = _TLS.conn = http.client.HTTPSConnection(HOST, timeout=120)
    conn.request("GET", PATH, headers={"Range": f"bytes={offset}-{offset + size - 1}"})
    resp = conn.getresponse()
    body = resp.read()
    if resp.status != 206:
        raise SystemExit(f"服务器返回 {resp.status}，不是 206")
    return len(body)


def cmd_rate(args) -> None:
    rows = []
    for n in args.threads:
        jobs = [(args.base + i * (args.size + 8192), args.size) for i in range(n * args.rounds)]
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=n) as pool:
            got = list(pool.map(lambda job: _fetch_into(*job), jobs))
        dt = time.time() - t0
        rows.append(
            {
                "threads": n,
                "requests": len(jobs),
                "req_per_s": round(len(jobs) / dt, 2),
                "KB_per_s": round(sum(got) / dt / 1024, 1),
                "seconds": round(dt, 1),
            }
        )
    print(json.dumps({"block_KB": args.size // 1024, "rows": rows}, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# cell：真实代码路径
# ---------------------------------------------------------------------------
def _manifest_entry(batch_no: int) -> dict:
    path = Path("data/mit-stanford/manifest.json")
    if not path.is_file():
        raise SystemExit("先跑 scripts/build_manifest.py 生成 manifest")
    manifest = json.loads(path.read_text("utf-8"))
    return next(e for e in manifest["batches"] if e["batch_no"] == batch_no)


def cmd_cell(args) -> None:
    import numpy as np  # noqa: F401 - build_features 间接需要

    import build_features as bf
    from build_manifest import resolve_source

    entry = _manifest_entry(args.batch)
    grid = bf.make_v_grid(bf.DEFAULT_V_LO, bf.DEFAULT_V_HI, bf.DEFAULT_POINTS)
    t0 = time.time()
    results = bf.extract_batch(
        entry,
        resolve_source(args.batch, None),
        cycles=bf.parse_cycles(bf.DEFAULT_CYCLES),
        ref_cycle=bf.DEFAULT_REF_CYCLE,
        grid=grid,
        limit=args.cells,
        workers=args.workers,
    )
    dt = time.time() - t0
    print(
        json.dumps(
            {
                "cells": len(results),
                "workers": args.workers,
                "seconds": round(dt, 1),
                "per_cell_s": round(dt / max(1, len(results)), 1),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# addrs：地址分布
# ---------------------------------------------------------------------------
def cmd_addrs(args) -> None:
    import numpy as np  # noqa: F401 - build_features 间接需要

    import build_features as bf
    from build_manifest import resolve_source
    from flsoh import matr

    spans: list[tuple[int, int]] = []
    orig = matr.HTTPRangeFile._fetch

    def spy(self, start, end):
        spans.append((start, end))
        return orig(self, start, end)

    matr.HTTPRangeFile._fetch = spy
    try:
        entry = _manifest_entry(args.batch)
        grid = bf.make_v_grid(bf.DEFAULT_V_LO, bf.DEFAULT_V_HI, bf.DEFAULT_POINTS)
        bf.extract_batch(
            entry,
            resolve_source(args.batch, None),
            cycles=bf.parse_cycles(bf.DEFAULT_CYCLES),
            ref_cycle=bf.DEFAULT_REF_CYCLE,
            grid=grid,
            limit=1,
            workers=1,
        )
    finally:
        matr.HTTPRangeFile._fetch = orig

    starts = sorted(s for s, _ in spans)
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    sizes = sorted(e - s for s, e in spans)
    total = sum(sizes)
    print(
        json.dumps(
            {
                "requests": len(spans),
                "bytes_KB": round(total / 1024),
                "median_bytes": sizes[len(sizes) // 2] if sizes else 0,
                "span_MB": round((starts[-1] - starts[0]) / 1024 / 1024, 1) if starts else 0,
                "gap_under_32KB_share": round(sum(1 for g in gaps if g < 32768) / max(1, len(gaps)), 3),
                "gap_under_128KB_share": round(sum(1 for g in gaps if g < 131072) / max(1, len(gaps)), 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rate = sub.add_parser("rate", help="纯 HTTP 层的请求速率与字节吞吐")
    p_rate.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8])
    p_rate.add_argument("--size", type=int, default=256 << 10)
    p_rate.add_argument("--rounds", type=int, default=4)
    p_rate.add_argument("--base", type=int, default=120_000_000)
    p_rate.set_defaults(func=cmd_rate)

    p_cell = sub.add_parser("cell", help="真实路径下抽 N 颗的墙钟时间")
    p_cell.add_argument("--cells", type=int, default=2)
    p_cell.add_argument("--workers", type=int, default=1)
    p_cell.add_argument("--batch", type=int, default=1)
    p_cell.set_defaults(func=cmd_cell)

    p_addr = sub.add_parser("addrs", help="单颗请求的地址分布")
    p_addr.add_argument("--batch", type=int, default=1)
    p_addr.set_defaults(func=cmd_addrs)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
