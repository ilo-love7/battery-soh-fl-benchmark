"""MATR（MIT-Stanford 电池老化数据集）的远端随机读取与解析。

数据出处
--------
canonical 源是 https://data.matr.io/1/ （需注册，学术免费）。本模块默认从一个
GitHub 镜像按字节区间随机读，原因是**整篇下载不可行**：三个 .mat 合计约 7.7 GiB，
单连接实测只有 17 KB/s；而本项目每颗电池真正需要的只有 summary 的 8 条曲线
（约 76 KB）加几十条充电曲线的 V/Qc，三个批次合计几 MB。为 0.06% 的字节付
7.7 GiB 的带宽没有道理。

MATLAB v7.3 就是 HDF5，HDF5 允许按区间随机读。所以这里实现一个「HTTP Range
支撑的可 seek 文件对象」，交给 h5py 的 ``fileobj`` 驱动。本地文件同样支持，
行为完全一致——先把文件下全了再跑，与直接远端读，走的是同一条代码路径。

电池 ID 的口径
--------------
文件里**没有**人类可读的电池名。``channel_id`` 是形如
``[3707764736, 2, 1, 1, 47, 1]`` 的 uint32 数组（第 5 位是通道号），
``barcode`` 是 1 起始的顺序号。社区惯例（rdbraatz 的 BuildPkl_*.ipynb）按位置
命名：``b1c{i}`` / ``b2c{i}`` / ``b3c{i}``，i 从 0 开始。本模块沿用该惯例，
并把 channel_id / barcode 一并写进 manifest 供回溯。

因此 cells.txt 是**生成物**，不是手写清单——手写清单会和真实数据脱节，
而脱节在精度曲线上看不出来。这正是 P-UNKNOWN 检查存在的理由。
"""

from __future__ import annotations

import http.client
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import h5py
import numpy as np

# summary 结构体的字段。顺序照 MATLAB 里的定义，不按字母排——打印时更容易
# 和原文对照。
SUMMARY_FIELDS = (
    "IR",
    "QCharge",
    "QDischarge",
    "Tavg",
    "Tmax",
    "Tmin",
    "chargetime",
    "cycle",
)

# 每个循环的曲线字段。本项目只用 V / Qc，其余留作完整版（1D-CNN）的输入。
CURVE_FIELDS = ("I", "Qc", "Qd", "Qdlin", "T", "Tdlin", "V", "discharge_dQdV", "t")

MATR_MIRROR = (
    "https://media.githubusercontent.com/media/DESN-Group-F/bsc-f2a5cc56/main/"
    "data/raw/original_sources/collection/raw/science/SRC-017"
)

# 论文用的三个批次。第 4 个批次（2019-01-24）来自 Attia 的闭环快充论文，
# 不属于 Severson 的 124 颗，所以不在这里。
BATCH_FILES = {
    1: "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    2: "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    3: "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
BATCH_BY_DATE = {name.split("_")[0]: no for no, name in BATCH_FILES.items()}


def mirror_url(batch_no: int) -> str:
    """批次号 → 镜像 URL。"""
    if batch_no not in BATCH_FILES:
        raise MatrError(f"未知批次号 {batch_no}；只有 {sorted(BATCH_FILES)}")
    return f"{MATR_MIRROR}/{BATCH_FILES[batch_no]}"


class MatrError(RuntimeError):
    """数据源、结构或契约不满足时抛出。不静默降级。"""


def _decode_text(arr) -> str:
    """MATLAB 的 char 数组在 HDF5 里是 uint16：ASCII 占一个字节、隔一个 \\x00。"""
    a = np.asarray(arr)
    if a.dtype.kind in ("U", "S"):
        return a.ravel()[0].item() if a.size else ""
    if a.dtype.kind != "u":
        raise MatrError(f"无法解码的文本类型 {a.dtype}")
    raw = a.astype("<u2" if a.dtype.itemsize == 2 else "u1").tobytes()
    if a.dtype.itemsize == 2:
        raw = raw[::2]
    return raw.split(b"\x00")[0].decode("ascii", "replace")


class HTTPRangeFile:
    """只读、可 seek 的文件对象：把 HTTP Range 请求伪装成本地文件。

    给 h5py 的 ``fileobj`` 驱动用。三处设计都不是可选项：

    1. **精确区间**。不做固定块缓存。HDF5 的元数据是几十到几百字节的碎片，
       用 1 MB 的块去接会浪费两个数量级（实测：1 MB 块下打开一个 3 GB 的
       远端文件要抓 4 MB，只为读出 6 个数字）。
    2. **并行补洞**。单连接实测 17 KB/s，瓶颈在每连接限速；6 个并发能到
       131 KB/s。所以一次 read 缺几段就并发抓几段。
    3. **拒绝降级**。服务器若忽略 Range 返回 200 整篇，直接抛错——那会把
       3 GB 拉进内存，而且是静默的。
    """

    def __init__(
        self,
        url: str,
        *,
        workers: int = 8,
        chunk: int = 2 << 20,
        timeout: int = 60,
        retry_delay: float = 0.6,
    ) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in ("https", "http"):
            raise MatrError(f"只接受 http(s) URL，收到 {url}")
        self.url = url
        self._secure = parsed.scheme == "https"
        self._host = parsed.netloc
        self._path = parsed.path + (("?" + parsed.query) if parsed.query else "")
        self._workers = max(1, int(workers))
        self._chunk = max(64 << 10, int(chunk))
        self._timeout = int(timeout)
        self._retry_delay = float(retry_delay)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._cache: list[tuple[int, bytes]] = []  # 有序、不重叠
        self.pos = 0
        self.requests = 0
        self.bytes_fetched = 0
        self.size = self._probe_size()

    # ------------------------------------------------------------------
    # 连接
    # ------------------------------------------------------------------
    def _conn(self) -> http.client.HTTPConnection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            factory = (
                http.client.HTTPSConnection
                if self._secure
                else http.client.HTTPConnection
            )
            conn = factory(self._host, timeout=self._timeout)
            self._local.conn = conn
        return conn

    def _reset(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        self._local.conn = None

    def _fetch(self, start: int, end: int) -> bytes:
        """抓 [start, end)。长度必须严丝合缝，短了就是错，不能凑。"""
        reason = ""
        for attempt in range(6):
            try:
                conn = self._conn()
                conn.request(
                    "GET", self._path, headers={"Range": f"bytes={start}-{end - 1}"}
                )
                resp = conn.getresponse()
                body = resp.read()
                if resp.status == 206:
                    if len(body) != end - start:
                        raise MatrError(
                            f"Range 长度不符：要 {end - start} B，实得 {len(body)} B"
                        )
                    with self._lock:
                        self.requests += 1
                        self.bytes_fetched += len(body)
                    return body
                if resp.status == 200:
                    raise MatrError(
                        "服务器忽略 Range 返回整篇；拒绝把整个文件拉进内存"
                    )
                reason = f"HTTP {resp.status}"
                self._reset()
            except MatrError:
                raise
            except (http.client.HTTPException, OSError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
                self._reset()
            time.sleep(self._retry_delay * (attempt + 1))
        raise MatrError(f"区间抓取失败 [{start}, {end})：{reason}")

    def _probe_size(self) -> int:
        for attempt in range(6):
            try:
                conn = self._conn()
                conn.request("GET", self._path, headers={"Range": "bytes=0-0"})
                resp = conn.getresponse()
                resp.read()
                content_range = resp.getheader("Content-Range") or ""
                if "/" in content_range:
                    return int(content_range.rsplit("/", 1)[1])
                self._reset()
            except (http.client.HTTPException, OSError):
                self._reset()
            time.sleep(self._retry_delay * (attempt + 1))
        raise MatrError("无法从 Content-Range 得知远端文件大小")

    # ------------------------------------------------------------------
    # 缓存
    # ------------------------------------------------------------------
    def _missing(self, start: int, end: int) -> list[tuple[int, int]]:
        """[start, end) 里还没缓存的子区间。"""
        gaps = [(start, end)]
        for cached_start, buf in self._cache:
            cached_end = cached_start + len(buf)
            if cached_end <= start or cached_start >= end:
                continue
            remaining: list[tuple[int, int]] = []
            for gap_start, gap_end in gaps:
                if gap_end <= cached_start or gap_start >= cached_end:
                    remaining.append((gap_start, gap_end))
                    continue
                if gap_start < cached_start:
                    remaining.append((gap_start, cached_start))
                if gap_end > cached_end:
                    remaining.append((cached_end, gap_end))
            gaps = remaining
            if not gaps:
                break
        return gaps

    def _insert(self, start: int, buf: bytes) -> None:
        with self._lock:
            self._cache.append((start, buf))
            self._cache.sort(key=lambda item: item[0])
            merged: list[tuple[int, bytes]] = []
            for item_start, item_buf in self._cache:
                if merged and item_start == merged[-1][0] + len(merged[-1][1]):
                    merged[-1] = (merged[-1][0], merged[-1][1] + item_buf)
                else:
                    merged.append((item_start, item_buf))
            self._cache = merged

    def _pull(self, ranges: list[tuple[int, int]]) -> None:
        """并发抓若干区间。大区间先切块，切块才能并行。"""
        pieces: list[tuple[int, int]] = []
        for start, end in ranges:
            while end - start > self._chunk:
                pieces.append((start, start + self._chunk))
                start += self._chunk
            if end > start:
                pieces.append((start, end))
        if not pieces:
            return
        if len(pieces) == 1:
            start, end = pieces[0]
            self._insert(start, self._fetch(start, end))
            return
        with ThreadPoolExecutor(max_workers=min(self._workers, len(pieces))) as pool:
            bufs = list(pool.map(lambda span: self._fetch(*span), pieces))
        for (start, _), buf in zip(pieces, bufs):
            self._insert(start, buf)

    def prefetch(self, ranges) -> None:
        """预取若干 [start, end)。之后 read 命中缓存，不再发请求。"""
        spans = [(int(start), int(end)) for start, end in ranges if end > start]
        if spans:
            self._pull(spans)

    # ------------------------------------------------------------------
    # 文件协议
    # ------------------------------------------------------------------
    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return True

    def closed(self) -> bool:
        return False

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0:
            return b""
        start, end = self.pos, min(self.pos + n, self.size)
        gaps = self._missing(start, end)
        if gaps:
            self._pull(gaps)
        out = bytearray()
        for cached_start, buf in self._cache:
            cached_end = cached_start + len(buf)
            if cached_end <= start:
                continue
            if cached_start >= end:
                break
            lo, hi = max(cached_start, start), min(cached_end, end)
            if hi > lo:
                out += buf[lo - cached_start : hi - cached_start]
        self.pos = end
        if len(out) != end - start:
            raise MatrError(
                f"读到的字节数不符：要 {end - start} B，得 {len(out)} B"
            )
        return bytes(out)

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        else:
            raise MatrError(f"不支持的 whence={whence}")
        return self.pos

    def tell(self) -> int:
        return self.pos

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self._reset()

    def __enter__(self) -> "HTTPRangeFile":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


@dataclass(frozen=True)
class CellMeta:
    """一颗电池的元信息。``cell_id`` 按位置生成，口径见模块文档。"""

    index: int
    batch_no: int
    cell_id: str
    channel_id: tuple[int, ...]
    barcode: int
    policy: str
    cycle_life: float
    n_cycles: int


def _data_span(dataset) -> tuple[int, int] | None:
    """数据集载荷的 [起始, 结束) 字节区间；非连续存储返回 None。

    chunked 数据集上 ``get_offset()`` 给的是块索引地址而不是数据地址，
    拿它去预取会抓回一堆无关字节，所以直接放弃预取——只影响速度，不影响正确性。
    """
    try:
        if dataset.chunks is not None:
            return None
        offset = dataset.id.get_offset()
        size = dataset.id.get_storage_size()
    except Exception:
        return None
    if offset is None or offset < 0 or size is None or size <= 0:
        return None
    return (int(offset), int(offset) + int(size))


class MatrBatch:
    """一个 .mat 批次的只读视图。

    生命周期：``MatrBatch(source)`` 打开 → ``cells`` / ``summary`` / ``curve``
    读取 → ``close()``。``source`` 可以是远端 https URL，也可以是本地路径。
    """

    def __init__(self, source: str | Path, *, batch_no: int | None = None) -> None:
        self.source = str(source)
        if self.source.startswith(("http://", "https://")):
            self.url: str | None = self.source
            self._file: object = HTTPRangeFile(self.source)
        else:
            self.url = None
            path = Path(self.source)
            if not path.is_file():
                raise MatrError(f"文件不存在：{path}")
            self._file = open(path, "rb")
        self._h5 = h5py.File(self._file, driver="fileobj")
        self._cell_cache: dict[int, CellMeta] = {}
        # 引用数组按字段**整批**读一次。逐格取 dataset[i, 0] 会让 h5py 为每个
        # 下标各发一个请求——46 颗电池就是 46 次往返。远端随机读的成本几乎全在
        # 往返上（实测每请求约 1 s），不在字节上，所以这条优化比缓存块大小重要。
        self._batch_refs: dict[str, np.ndarray] = {}
        self._summary_groups: dict[int, object] = {}
        self._cycle_refs_cache: dict[tuple[int, str], np.ndarray] = {}
        try:
            self._load(batch_no)
        except Exception:
            self.close()
            raise

    # ------------------------------------------------------------------
    # 打开与结构校验
    # ------------------------------------------------------------------
    def _load(self, batch_no: int | None) -> None:
        if "batch" not in self._h5:
            raise MatrError(f"{self.source} 顶层没有 batch 变量，不是 MATR 的 .mat")
        raw_date = ""
        if "batch_date" in self._h5:
            raw_date = _decode_text(self._h5["batch_date"][()])
        self.batch_date = raw_date
        if batch_no is None:
            batch_no = BATCH_BY_DATE.get(raw_date) or BATCH_BY_DATE.get(
                Path(self.source).name.split("_")[0]
            )
        if batch_no is None:
            raise MatrError(
                f"无法判断批次号（batch_date={raw_date!r}）；请显式传 batch_no，"
                "否则 cell_id 的 b{批次}c{位置} 口径无从生成"
            )
        self.batch_no = int(batch_no)
        self._batch = self._h5["batch"]
        if "summary" not in self._batch:
            raise MatrError("batch 缺少 summary 字段")
        self.n_cells = int(self._batch["summary"].shape[0])
        if self.n_cells <= 0:
            raise MatrError("batch/summary 为空")
        # 各字段长度必须一致：不一致说明读错了文件，或者该文件是拼接产物。
        # 按下标硬取会静默错位，错位后的指标看起来完全正常。
        for field in (
            "summary",
            "cycles",
            "cycle_life",
            "channel_id",
            "barcode",
            "policy_readable",
        ):
            if field not in self._batch:
                raise MatrError(f"batch 缺少 {field} 字段")
            n = int(self._batch[field].shape[0])
            if n != self.n_cells:
                raise MatrError(
                    f"batch/{field} 有 {n} 项，与 summary 的 {self.n_cells} 项不一致"
                )

    # ------------------------------------------------------------------
    # 引用解引用
    # ------------------------------------------------------------------
    def _batch_ref(self, field: str) -> np.ndarray:
        """batch 下某字段的引用数组，整批读一次后缓存。"""
        refs = self._batch_refs.get(field)
        if refs is None:
            refs = np.asarray(self._batch[field][()]).reshape(-1)
            self._batch_refs[field] = refs
        return refs

    def _ref_object(self, field: str, index: int):
        refs = self._batch_ref(field)
        if not 0 <= index < len(refs):
            raise MatrError(f"电池下标 {index} 越界（{field} 共 {len(refs)} 项）")
        return self._h5[refs[index]]

    def _summary_group(self, index: int):
        group = self._summary_groups.get(index)
        if group is None:
            group = self._ref_object("summary", index)
            missing = [f for f in SUMMARY_FIELDS if f not in group]
            if missing:
                raise MatrError(f"电池 {index} 的 summary 缺字段 {missing}")
            self._summary_groups[index] = group
        return group

    def _cycle_refs(self, index: int, field: str) -> np.ndarray:
        """该电池某通道的逐循环引用数组，整批读一次后缓存。"""
        key = (index, field)
        refs = self._cycle_refs_cache.get(key)
        if refs is None:
            group = self._ref_object("cycles", index)
            if field not in group:
                raise MatrError(
                    f"电池 {index} 的 cycles 缺字段 {field}（有 {list(group.keys())}）"
                )
            refs = np.asarray(group[field][()]).reshape(-1)
            self._cycle_refs_cache[key] = refs
        return refs

    # ------------------------------------------------------------------
    # 公开读取接口
    # ------------------------------------------------------------------
    @property
    def cells(self) -> list[CellMeta]:
        return [self.cell(i) for i in range(self.n_cells)]

    def cell(self, index: int) -> CellMeta:
        """单颗电池的元信息。按需读、读后缓存——并发抽取时每个 worker 只碰自己那几颗。"""
        meta = self._cell_cache.get(index)
        if meta is None:
            if not 0 <= index < self.n_cells:
                raise MatrError(f"电池下标 {index} 越界（共 {self.n_cells} 颗）")
            meta = self._read_cell(index)
            self._cell_cache[index] = meta
        return meta

    def policy(self, index: int) -> str:
        """只要 ``policy_readable`` 这一条。

        比 ``cell()`` 少 4 次往返（channel_id / barcode / cycle_life / summary
        shape）。验证跨批次延续关系要读整批 46 颗，那 4 次乘 46 就是三分钟——
        远端读的成本全在往返上，所以「少读一条」值得单独开个方法。
        """
        return _decode_text(self._ref_object("policy_readable", index)[()])

    def _read_cell(self, index: int) -> CellMeta:
        channel = np.asarray(self._ref_object("channel_id", index)[()]).ravel()
        barcode = np.asarray(self._ref_object("barcode", index)[()]).ravel()
        policy = _decode_text(self._ref_object("policy_readable", index)[()])
        cycle_life = np.asarray(self._ref_object("cycle_life", index)[()]).ravel()
        n_cycles = int(self._summary_group(index)["QDischarge"].shape[1])
        return CellMeta(
            index=index,
            batch_no=self.batch_no,
            cell_id=f"b{self.batch_no}c{index}",
            channel_id=tuple(int(x) for x in channel),
            barcode=int(barcode[0]) if barcode.size else -1,
            policy=policy,
            cycle_life=float(cycle_life[0]) if cycle_life.size else float("nan"),
            n_cycles=n_cycles,
        )

    def summary(self, index: int, fields=None) -> dict[str, np.ndarray]:
        """该电池的逐循环标量曲线。默认全读；``fields`` 可只取需要的几条。

        只取 ``QDischarge`` 与 ``cycle`` 能省 6 次往返——远端读的成本在往返上，
        所以这个参数不是装饰。
        """
        group = self._summary_group(index)
        wanted = tuple(SUMMARY_FIELDS if fields is None else fields)
        unknown = [f for f in wanted if f not in SUMMARY_FIELDS]
        if unknown:
            raise MatrError(f"未知 summary 字段 {unknown}；只有 {SUMMARY_FIELDS}")
        out: dict[str, np.ndarray] = {}
        for field in wanted:
            out[field] = np.asarray(group[field][()], dtype=float).ravel()
        n = len(out["QDischarge"]) if "QDischarge" in out else len(next(iter(out.values())))
        for field, values in out.items():
            if len(values) != n:
                raise MatrError(
                    f"电池 {index} 的 summary 字段长度不一：{field}={len(values)} "
                    f"vs {n}"
                )
        return out

    def curve(self, index: int, cycle: int, field: str = "V") -> np.ndarray:
        """第 ``cycle`` 个循环的某个通道（0 起，与 summary 的下标对齐）。"""
        if field not in CURVE_FIELDS:
            raise MatrError(f"未知曲线字段 {field}；只有 {CURVE_FIELDS}")
        refs = self._cycle_refs(index, field)
        if not 0 <= cycle < len(refs):
            raise MatrError(f"循环下标 {cycle} 越界（电池 {index} 共 {len(refs)} 个）")
        return np.asarray(self._h5[refs[cycle]][()], dtype=float).ravel()

    def charge_curve(self, index: int, cycle: int) -> tuple[np.ndarray, np.ndarray]:
        """(V, Qc)。两者长度必须一致，否则后面插值会错位。"""
        voltage = self.curve(index, cycle, "V")
        capacity = self.curve(index, cycle, "Qc")
        if voltage.shape != capacity.shape:
            raise MatrError(
                f"电池 {index} 循环 {cycle} 的 V/Qc 长度不一致："
                f"{voltage.shape} vs {capacity.shape}"
            )
        return voltage, capacity

    def n_cycles(self, index: int) -> int:
        return int(len(self._cycle_refs(index, "V")))

    # ------------------------------------------------------------------
    # 预取
    # ------------------------------------------------------------------
    def prefetch_cell(self, index: int, cycles, fields=("V", "Qc")) -> int:
        """把该电池本次要用的数据区间并发抓回缓存，返回预取的字节数。

        做法是**包络预取**：先读首尾两个循环的对象头拿到字节偏移，再按
        ``[最小偏移, 最大偏移)`` 一次抓回。MATLAB 写 HDF5 时同一颗电池的循环
        是顺序分配的，所以包络内的浪费有限，而请求数从「每循环每通道一次」
        降到一次。

        包络不成立（首尾乱序，或跨度大到说明布局不像预期）时退化为逐个预取。
        这只影响速度，不影响正确性——真正的读取仍然走正常的缓存/抓取路径。
        """
        spans: list[tuple[int, int]] = []
        group = self._summary_group(index)
        for field in SUMMARY_FIELDS:
            span = _data_span(group[field])
            if span:
                spans.append(span)

        for field in fields:
            refs = self._cycle_refs(index, field)
            wanted = [c for c in cycles if 0 <= c < len(refs)]
            if not wanted:
                continue
            first = _data_span(self._h5[refs[wanted[0]]])
            last = _data_span(self._h5[refs[wanted[-1]]])
            if first and last and first[0] < last[0]:
                envelope = (first[0], last[1])
                if envelope[1] - envelope[0] <= 32 << 20:
                    spans.append(envelope)
                    continue
            for cycle in wanted:
                span = _data_span(self._h5[refs[cycle]])
                if span:
                    spans.append(span)

        total = sum(end - start for start, end in spans)
        if isinstance(self._file, HTTPRangeFile) and spans:
            self._file.prefetch(spans)
        return total

    @property
    def requests(self) -> int:
        """已发出的 HTTP 请求数。本地文件恒为 0。"""
        return int(getattr(self._file, "requests", 0))

    @property
    def bytes_fetched(self) -> int:
        """已下载的字节数。本地文件恒为 0。"""
        return int(getattr(self._file, "bytes_fetched", 0))

    @property
    def n_bytes(self) -> int | None:
        """源文件大小。远端读时来自 Content-Range，本地读时来自 stat。"""
        size = getattr(self._file, "size", None)
        if size is not None:
            return int(size)
        try:
            return Path(self.source).stat().st_size
        except OSError:
            return None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def close(self) -> None:
        try:
            self._h5.close()
        finally:
            self._file.close()

    def __enter__(self) -> "MatrBatch":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"MatrBatch(batch={self.batch_no}, date={self.batch_date!r}, "
            f"cells={self.n_cells})"
        )
