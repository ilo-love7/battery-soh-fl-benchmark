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
    2. **并行补洞，且连接要活过这一次调用**。一次 read 缺几段就并发抓几段。
       并发在这里**几乎用不上**（见下），但「连接不能每次重建」是硬要求。
    3. **拒绝降级**。服务器若忽略 Range 返回 200 整篇，直接抛错——那会把
       3 GB 拉进内存，而且是静默的。

    **一个真 bug：每批请求都重新握手**

    早期 ``_pull`` 每次都新建 ``ThreadPoolExecutor``，``with`` 退出时
    ``shutdown(wait=True)`` 把线程 join 掉，线程本地的连接随之失效。
    于是**每一批请求都重做一次 TLS 握手**。量出来像是「镜像惩罚并发」：

    ============ ==================== ==================== ==============
    并发数        前 2 颗耗时           单颗稳态耗时          当时的结论
    ============ ==================== ==================== ==============
    1             60 s                 约 24 s              基准
    2             253 s                —                    更慢
    6             404 s（首颗）        约 100 s             更慢
    ============ ==================== ==================== ==============

    据此把默认值改成 1，还写了「并发有害，别改回去」——**那个结论是错的**。
    拆开量同一主机、同一时段：

    ======================== =============== ============== ==============
    场景                      请求速率        字节吞吐        相对单流
    ======================== =============== ============== ==============
    每次新建连接、1 线程        0.63 个/秒      13.8 KB/s      1.0×
    复用连接、1 线程            2.56 个/秒      —              4.1×（请求）
    复用连接、8 线程            8.45 个/秒      76.2 KB/s      5.5×（字节）
    复用连接、12 线程           —               115.5 KB/s     8.4×
    复用连接、24 线程           —               132.9 KB/s     9.6×
    ======================== =============== ============== ==============

    修法：``_pool`` 只建一次、跨 ``_pull`` 复用，线程不销毁，线程本地连接
    就一直活着。真实数据上量到的收益是 **1.39 倍**（同小时、各 2 颗：
    128.9 s/颗 → 92.5 s/颗）。

    **但上面那张 5.5 倍的表不要拿来当预期收益，它量错了对象**

    那张表是 256 KB **大块**传输的吞吐，而真实负载是 134 个**中位 512 字节**
    的小请求。单颗诊断（132 次 ``_pull``、134 次抓取、427 KB）：

    - 只有 **2 次** ``_pull`` 有多个缺口（1.5%）——所以内部并发基本不触发，
      这才是 8 线程只换来 1.39 倍而不是 5.5 倍的原因；
    - 抓取耗时占总墙钟 **98.2%**，字节只占 427 KB——**纯往返延迟受限**。

    **颗级并行也试过，同样无效**（4 颗一组，同小时）：

    ============ =============== ================
    顶层 workers   单颗耗时         相对串行
    ============ =============== ================
    1             120.5 s         1.00×
    2             127.8 s         0.94×
    4             106.1 s         1.14×
    ============ =============== ================

    颗与颗之间确实没有依赖，所以问题不在依赖，在**服务端**：它对持续请求
    速率有聚合节流，约 **1.5 次/秒**，与开多少连接无关。134 次 ÷ 1.5 ≈ 89 s，
    正好等于实测的 89.7 s/颗。旁证：热连接上单独发一个 512 B 请求只要
    **0.17 s**，而真实负载里同样的请求要 0.64 s——短突发不限、持续流限。

    所以**预读窗口**也没戏。离线回放真实地址序列（地址确实聚簇，93% 的
    相邻间隔 < 128 KB）：

    ============ ========== ========== ==========
    窗口          请求数      字节        字节倍数
    ============ ========== ========== ==========
    0             240        842 KB     1.00×
    16 KB         113        1808 KB    2.15×
    64 KB         67         4288 KB    5.09×
    ============ ========== ========== ==========

    窗口 16 KB 能把请求砍掉 53%，但字节放大 2.15 倍；而字节吞吐只有约
    13 KB/s（实测 8 KB 0.59 s、64 KB 5.05 s），多出来的字节按 904 KB/颗
    算要 69 s，比省下的 36 s 往返更贵。**省得少、费得多，方向反了。**

    结论：**这一层已经没有软件杠杆了**。能做的只有少抓字节（已贴着
    每循环每通道一次对象头的下限）和别浪费连接（已修）。剩下的时间是
    镜像的节流，不是我们的代码。顺带记一个运营事实：**同一个镜像的吞吐
    按时段差 5 倍**（早上约 24 s/颗，晚间约 120 s/颗），所以任何吞吐数字
    都必须带上测量的时段，否则下一个读的人会当成常量。
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
        # 连接登记表：线程本地的连接没法从别的线程拿到，close() 时需要一个
        # 能遍历的清单，否则池里的连接会一直挂到进程退出。
        self._conns: list[http.client.HTTPConnection] = []
        # 线程池只建一次。每次 _pull 新建一个会在退出时 join 掉线程，
        # 线程本地的连接随之失效，下一批请求又付一次 TLS 握手——实测这就是
        # 「并发看起来有害」的真正原因，见类 docstring 的实测表。
        self._pool: ThreadPoolExecutor | None = None
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
            with self._lock:
                self._conns.append(conn)
        return conn

    def _executor(self) -> ThreadPoolExecutor:
        """跨 ``_pull`` 复用的线程池。线程活着，连接才活着。"""
        pool = self._pool
        if pool is None:
            with self._lock:
                if self._pool is None:
                    self._pool = ThreadPoolExecutor(
                        max_workers=self._workers,
                        thread_name_prefix="http-range",
                    )
                pool = self._pool
        return pool

    def _reset(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            with self._lock:
                try:
                    self._conns.remove(conn)
                except ValueError:
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
        if len(pieces) == 1 or self._workers == 1:
            # 单段不绕线程池：调用方线程的连接本来就是常驻的。
            for start, end in pieces:
                self._insert(start, self._fetch(start, end))
            return
        bufs = list(self._executor().map(lambda span: self._fetch(*span), pieces))
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
        pool, self._pool = self._pool, None
        if pool is not None:
            # wait=True：等手上那一批请求收尾，别让线程往已关闭的连接上写。
            try:
                pool.shutdown(wait=True, cancel_futures=True)
            except TypeError:  # pragma: no cover - 3.8 及更早没有 cancel_futures
                pool.shutdown(wait=True)
        with self._lock:
            conns, self._conns = self._conns, []
        for conn in conns:
            try:
                conn.close()
            except Exception:
                pass
        self._local.conn = None

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
        # 包络预取为什么没在用。省请求（146→73）但费字节（415 KB→2.0 MB），
        # 而字节才是这个镜像的瓶颈，方向反了。理由见 prefetch_cell 的 docstring。
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
                     请求数              字节数              旧实现下的耗时
        ============ =================== ================== ==============
        包络          73                  **2.0 MB**          约 90 s
        逐个解引用    **125**             415 KB              约 59 s
        ============ =================== ================== ==============

        「耗时」那列是在**连接不复用**的旧实现下量的（见类 docstring），
        绝对值已经不能直接引用了，但结论不变，理由换成两个上限的算术。

        服务端对持续请求有**聚合节流约 1.5 次/秒**（与连接数无关，见类
        docstring 的实测），字节吞吐约 **13 KB/s**。于是每颗的耗时近似
        ``max(请求数 / 1.5, 字节 / 13 KB/s)``：

        ============ ============ ============== ============== ==========
                     请求数        字节            请求项          字节项
        ============ ============ ============== ============== ==========
        包络          73           2.0 MB          49 s            **158 s**
        逐个解引用    **125**      415 KB          **83 s**        32 s
        ============ ============ ============== ============== ==========

        包络省请求只省到 0.58×，字节却放大到 4.8×——它把瓶颈从「请求」
        挪到了「字节」，而 158 s > 83 s。**省得少、费得多，方向是反的。**

        省得少、费得多，方向就是反的。反过来看：解引用一次约 2.3 个请求、
        4.2 KB，它已经贴着「每循环每通道一次对象头」的物理下限，没有中间地带。

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
