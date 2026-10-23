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


class MatrTransportError(MatrError):
    """**网络或远端出了问题**，与数据本身无关。

    单独一类是因为这两种失败的处置完全相反：

    - 数据问题（曲线不存在、循环越界、容量非正）→ 记进 ``dropped``，
      这颗电池算跑完，特征表里少几行，spec 里有据可查。
    - 传输问题（DNS 失败、连接重置、HTTP 5xx、服务器忽略 Range）→
      **不是数据缺失**。把它记进 ``dropped`` 等于把一次网络抖动
      写成一条数据缺陷：spec 里会躺着几十条「参考曲线读不出」，
      读起来就是"这些电池没有数据"。若这一轮没崩，命令正常退出、
      表正常写出，下游拿到一张少三分之一样本的表，唯一的告警是
      一段被误分类的丢弃原因。

    所以传输错误必须往上抛，让这一轮大声失败，重跑同一条命令即可。
    """


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

    **并发到底有没有用，要分指标看**（后来量出来的，上面第 2 条只对了一半）：

    - 请求数：聚合上限约 2.5 个/秒，单线程和 6 并发一样。加并发不省请求。
    - 字节数：聚合上限约 20 KB/s，同样与并发基本无关。

    所以「6 个并发 131 KB/s」那个数只在特定时段成立。真正省时间的是
    **少发请求、少抓字节**，不是多加连接。见
    :meth:`MatrBatch.prefetch_cell` 的实测对比表。
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
                    raise MatrTransportError(
                        "服务器忽略 Range 返回整篇；拒绝把整个文件拉进内存"
                    )
                reason = f"HTTP {resp.status}"
                self._reset()
            except MatrTransportError:
                raise
            except MatrError:
                raise
            except (http.client.HTTPException, OSError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
                self._reset()
            time.sleep(self._retry_delay * (attempt + 1))
        raise MatrTransportError(f"区间抓取失败 [{start}, {end})：{reason}")

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
        raise MatrTransportError("无法从 Content-Range 得知远端文件大小")

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
    """数据集载荷的 [起始, 结束) 字节区间；取不到就返回 None。

    ``DatasetID.get_offset()`` 对 MATLAB v7.3 写出来的这些数据集**一律返回
    None**——实测包括 contiguous 的 summary 字段（``chunks is None`` 但
    ``get_offset() is None``）。所以本函数基本用不上，保留它只是为了在
    真能取到时少走一层。

    真正管用的是 :func:`_object_addr`——对象头地址是能拿到的。
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


def _object_addr(obj) -> int | None:
    """任意 HDF5 对象（数据集或组）在文件里的字节地址。

    ``h5o.get_info().addr`` 对 contiguous / compact / chunked 都有效，
    而 ``get_offset()`` 只对 contiguous 且已分配的情况有效——MATLAB v7.3
    的数据集两边都不满足，所以前面那版包络预取从来没成立过。
    """
    try:
        return int(h5py.h5o.get_info(obj.id).addr)
    except Exception:
        return None


def _object_span(obj) -> tuple[int, int] | None:
    """对象的 [起始, 结束) 字节区间。结束地址按 storage size 估，是上界。"""
    addr = _object_addr(obj)
    if addr is None or addr < 0:
        return None
    try:
        size = int(obj.id.get_storage_size())
    except Exception:
        size = 0
    if size <= 0:
        size = 0
    return (addr, addr + size)


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
        # 包络预取为什么没在用。服务端聚合吞吐约 2.5 请求/秒、约 20 KB/s，
        # 两个上限都量过；包络省请求却费字节，实测反而慢一倍。
        # 理由见 prefetch_cell 的 docstring，别让人以为这条路还在跑。
        self.prefetch_degraded: set[str] = set()
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
        """把该电池本次要用的曲线解引用一遍，让数据和对象头进缓存。

        **为什么不用包络预取**（一次抓一大段字节区间）：实测过，而且慢一倍。

        ``DatasetID.get_offset()`` 对这些 MATLAB v7.3 数据集一律返回 None，
        所以只能走 ``h5o.get_info().addr`` 拿对象头地址。地址确实单调递增
        （cycle1=42,358,623 → cycle100=43,276,349，每循环约 9,184 B），
        包络能成立。但代价是字节：

        ============ =================== ================== ==============
                     请求数              字节数              单颗耗时
        ============ =================== ================== ==============
        包络          73                  **2.0 MB**          约 90 s
        逐个解引用    **125**             415 KB              约 59 s
        ============ =================== ================== ==============

        服务端两个上限都量过：**聚合约 2.5 请求/秒**、**聚合约 20 KB/s**。
        包络把请求数砍半，却把字节数放大 5 倍，于是从「请求受限」变成
        「字节受限」，反而更慢。解引用一次约 2.3 个请求、4.2 KB——
        它就是这件事的最优解，没有中间地带。

        另一个关键事实：**解引用一次就等于读完整条曲线**。对象头里带着数据，
        所以解引用完再 ``charge_curve`` 是 0 请求 0 字节。这也意味着
        「先算 span 再决定怎么抓」那套逻辑全是白费——算 span 的那 21 次
        解引用已经把数据拿回来了。

        包络代码留在 :func:`_object_span` / :func:`_data_span`，换存储布局时
        可能又划得来；``prefetch_degraded`` 记下当前没用包络，别让人以为
        这条路径还在跑。
        """
        count = 0
        for field in fields:
            refs = self._cycle_refs(index, field)
            for cycle in cycles:
                if 0 <= cycle < len(refs):
                    self._h5[refs[cycle]]
                    count += 1
        self.prefetch_degraded.add(f"{fields}：未用包络，实测字节更省")
        return count

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
