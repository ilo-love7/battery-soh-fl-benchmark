"""``matr.py`` 的测试：合成 HDF5 + 本地 Range 服务器，全程不联网。

为什么值得为「远端随机读」写测试
--------------------------------
这类代码有三种**不会抛异常**的失效方式：

1. 服务器忽略 Range 返回 200 整篇 —— 3 GB 被静默拉进内存；
2. 区间长度与请求不符 —— 后续所有偏移整体错位；
3. 缓存拼接漏字节 —— 读出来长度正常、内容是垃圾。

三种都只在结果上体现为「数字有点怪」，而数字有点怪在精度曲线上看不出来。
所以这里起一个本地 HTTP 服务器，分别验证「支持 Range」与「忽略 Range」两条
路径，并用合成 HDF5 验证解析逻辑。
"""

from __future__ import annotations

import http.server
import threading

import pytest

# importorskip 必须在任何第三方 import 之前——写在后面的话，纯 pytest 的
# CI job（不装 numpy/h5py）会在 collection 阶段直接 ModuleNotFoundError，
# 得到「3 errors」而不是「全部 skipped」。这个坑踩过一次，别再踩。
h5py = pytest.importorskip("h5py", reason="matr 需要 h5py")
np = pytest.importorskip("numpy", reason="matr 需要 numpy")

from flsoh.matr import (  # noqa: E402
    SUMMARY_FIELDS,
    CellMeta,
    HTTPRangeFile,
    MatrBatch,
    MatrError,
    mirror_url,
)


# ---------------------------------------------------------------------------
# 合成一个 MATR 形状的 HDF5
# ---------------------------------------------------------------------------
def _char_array(text: str) -> np.ndarray:
    """MATLAB 的 char 在 HDF5 里是 uint16。"""
    return np.array([[ord(ch) for ch in text]], dtype=np.uint16)


def _ref_column(refs) -> np.ndarray:
    return np.array(list(refs), dtype=h5py.ref_dtype).reshape(-1, 1)


def write_fake_matr(
    path,
    *,
    n_cells: int = 3,
    n_cycles: int = 8,
    n_points: int = 32,
    date: str = "2017-05-12",
    drop_cycles: int = 0,
):
    """写一个结构与 MATR .mat 相同的 HDF5。

    ``drop_cycles`` 用来制造「字段长度不一致」的坏文件，验证结构校验会拦住它。
    """
    with h5py.File(path, "w") as h5:
        h5.create_dataset("batch_date", data=_char_array(date))
        batch = h5.create_group("batch")
        summary_refs, cycles_refs = [], []
        life_refs, channel_refs, barcode_refs, policy_refs = [], [], [], []

        for i in range(n_cells):
            base = 1.10 - 0.005 * np.arange(n_cycles, dtype=float)
            summary = h5.create_group(f"/fake/c{i}/summary")
            for field in SUMMARY_FIELDS:
                if field == "QDischarge":
                    values = base
                elif field == "cycle":
                    values = np.arange(1, n_cycles + 1, dtype=float)
                else:
                    values = np.ones(n_cycles)
                summary.create_dataset(field, data=values.reshape(1, -1))
            summary_refs.append(summary.ref)

            cycles = h5.create_group(f"/fake/c{i}/cycles")
            # 每个循环写成一整条「充电 + 放电」，与真实 MATR 同构：
            # 充电段 I>0、Qc 从 0 累积到 base；放电段 I<0、Qc 恒定、
            # Qd 从 0 累积到 base。**必须同构**——2026-10-25 那个 bug 的
            # 前提就是「循环数组里有两段」，只有一段的假数据测不出分段逻辑。
            half = n_points // 2
            v_refs, q_refs, d_refs, i_refs = [], [], [], []
            for k in range(n_cycles):
                v_chg = np.linspace(2.0, 3.6, half)
                v_dis = np.linspace(3.6, 2.0, n_points - half)
                qc = np.concatenate([
                    np.linspace(0.0, float(base[k]), half),
                    np.full(n_points - half, float(base[k])),
                ])
                qd = np.concatenate([
                    np.zeros(half),
                    np.linspace(0.0, float(base[k]), n_points - half),
                ])
                cur = np.concatenate([
                    np.full(half, 1.0),
                    np.full(n_points - half, -4.0),
                ])
                voltage = np.concatenate([v_chg, v_dis])
                v_refs.append(h5.create_dataset(f"/fake/c{i}/v{k}", data=voltage).ref)
                q_refs.append(h5.create_dataset(f"/fake/c{i}/q{k}", data=qc).ref)
                d_refs.append(h5.create_dataset(f"/fake/c{i}/d{k}", data=qd).ref)
                i_refs.append(h5.create_dataset(f"/fake/c{i}/i{k}", data=cur).ref)
            cycles.create_dataset("V", data=_ref_column(v_refs))
            cycles.create_dataset("Qc", data=_ref_column(q_refs))
            cycles.create_dataset("Qd", data=_ref_column(d_refs))
            cycles.create_dataset("I", data=_ref_column(i_refs))
            cycles_refs.append(cycles.ref)

            life_refs.append(
                h5.create_dataset(f"/fake/c{i}/life", data=np.float64(1000 + i)).ref
            )
            channel_refs.append(
                h5.create_dataset(
                    f"/fake/c{i}/ch",
                    data=np.array([[3707764736, 2, 1, 1, 47 + i, 1]], dtype=np.uint32),
                ).ref
            )
            barcode_refs.append(
                h5.create_dataset(
                    f"/fake/c{i}/bc", data=np.array([[i + 1]], dtype=np.uint32)
                ).ref
            )
            policy_refs.append(
                h5.create_dataset(f"/fake/c{i}/pol", data=_char_array("3.6C(80%)")).ref
            )

        batch.create_dataset("summary", data=_ref_column(summary_refs))
        batch.create_dataset("cycles", data=_ref_column(cycles_refs))
        batch.create_dataset("cycle_life", data=_ref_column(life_refs))
        batch.create_dataset("channel_id", data=_ref_column(channel_refs))
        batch.create_dataset("barcode", data=_ref_column(barcode_refs))
        batch.create_dataset("policy_readable", data=_ref_column(policy_refs))
        if drop_cycles:
            del batch["cycles"]
            batch.create_dataset("cycles", data=_ref_column(cycles_refs[:-drop_cycles]))


@pytest.fixture
def fake_mat(tmp_path):
    path = tmp_path / "2017-05-12_fake.mat"
    write_fake_matr(path)
    return path


# ---------------------------------------------------------------------------
# MatrBatch：结构解析
# ---------------------------------------------------------------------------
class TestMatrBatch:
    def test_reads_cells(self, fake_mat):
        with MatrBatch(fake_mat) as batch:
            assert batch.n_cells == 3
            assert batch.batch_no == 1
            assert [c.cell_id for c in batch.cells] == ["b1c0", "b1c1", "b1c2"]

    def test_cell_metadata(self, fake_mat):
        with MatrBatch(fake_mat) as batch:
            meta = batch.cell(1)
            assert isinstance(meta, CellMeta)
            assert meta.cell_id == "b1c1"
            assert meta.channel_id == (3707764736, 2, 1, 1, 48, 1)
            assert meta.barcode == 2
            assert meta.policy == "3.6C(80%)"
            assert meta.cycle_life == 1001.0
            assert meta.n_cycles == 8

    def test_summary_arrays(self, fake_mat):
        with MatrBatch(fake_mat) as batch:
            summary = batch.summary(0)
            assert set(summary) == set(SUMMARY_FIELDS)
            assert len(summary["QDischarge"]) == 8
            assert summary["cycle"][0] == 1.0
            # 衰减趋势：末循环容量低于首循环
            assert summary["QDischarge"][-1] < summary["QDischarge"][0]

    def test_curve_roundtrip(self, fake_mat):
        """``curve()`` 返回**整条循环**（原始通道），这是分段的输入。"""
        with MatrBatch(fake_mat) as batch:
            voltage, capacity, current = (
                batch.curve(2, 3, "V"), batch.curve(2, 3, "Qc"), batch.curve(2, 3, "I")
            )
            assert voltage.shape == (32,)
            assert capacity.shape == (32,)
            assert current.shape == (32,)
            # 整条循环：电压先升（充电）后降（放电）
            assert voltage[0] == pytest.approx(2.0)
            assert voltage[15] == pytest.approx(3.6)
            assert voltage[-1] == pytest.approx(2.0)
            assert (current[:16] > 0).all() and (current[16:] < 0).all()

    def test_discharge_curve_takes_only_the_discharge_segment(self, fake_mat):
        """``discharge_curve`` 必须只取 ``I<0`` 的那半段。

        这是 2026-10-25 那个 bug 的守门测试：曾经 ``charge_curve`` 返回整条
        循环，``delta_q_curve`` 的 ``argsort`` 于是把放电段排进低电压区，
        而放电段的 ``Qc`` 是常数——插出来的「ΔQ」总变差 269.6，正确值 0.21，
        两者相关系数 −0.02。

        注意返回的 V 是**升序**的（``segment_capacity_curve`` 排过）：断言不写
        「单调下降」，那按的是原始时间顺序，不是这个函数的契约。
        """
        with MatrBatch(fake_mat) as batch:
            v, q = batch.discharge_curve(2, 3)
            assert v.shape == (16,)                  # 只有 I<0 的半段
            assert (np.diff(v) > 0).all()            # 已按升序排好
            # 升序后：低压端是放电末尾（Qd 已充满），高压端是放电起点（Qd=0）
            assert q[0] > 0 and q[-1] == pytest.approx(0.0)

    def test_charge_curve_takes_only_the_charge_segment(self, fake_mat):
        with MatrBatch(fake_mat) as batch:
            v, q = batch.charge_curve(2, 3)
            assert v.shape == (16,)                  # 只有 I>0 的半段
            assert (np.diff(v) > 0).all()
            # 升序后：低压端是充电起点（Qc=0），高压端是充电末尾（Qc 已充满）
            assert q[0] == pytest.approx(0.0) and q[-1] > 0

    def test_segment_curves_do_not_overlap(self, fake_mat):
        """两段加起来必须恰好等于整条循环的点数——不重不漏。"""
        with MatrBatch(fake_mat) as batch:
            total = len(batch.curve(2, 3, "V"))
            vc, _ = batch.charge_curve(2, 3)
            vd, _ = batch.discharge_curve(2, 3)
            assert len(vc) + len(vd) == total

    def test_whole_cycle_is_not_a_valid_curve_input(self, fake_mat):
        """把整条循环喂给 ``delta_q_curve`` 必须被总变差判据抓住。

        这条是那个 bug 的**回归测试**：不 mock 任何东西，就用假 MATR 的同构
        数据，走一遍真实调用链。分段后 TV/幅度约 2（合成数据是线性的，所以
        比真实数据还小），不分段则大两个量级以上。
        """
        from flsoh.features import curve_total_variation, delta_q_curve, make_v_grid

        grid = make_v_grid(2.05, 3.55, 200)
        with MatrBatch(fake_mat) as batch:
            seg_ref = batch.discharge_curve(0, 1)
            seg_cur = batch.discharge_curve(0, 5)
            whole_ref = (batch.curve(0, 1, "V"), batch.curve(0, 1, "Qd"))
            whole_cur = (batch.curve(0, 5, "V"), batch.curve(0, 5, "Qd"))

        def _tv_ratio(ref, cur):
            d = delta_q_curve(*cur, grid) - delta_q_curve(*ref, grid)
            span = float(d.max() - d.min())
            return curve_total_variation(d) / span if span > 0 else float("inf")

        good = _tv_ratio(seg_ref, seg_cur)
        bad = _tv_ratio(whole_ref, whole_cur)
        assert good < 20.0, f"分段后的 TV/幅度 = {good:.1f}，本应远小于 20"
        assert bad > 5 * good, f"不分段 TV/幅度 {bad:.1f} 没有显著大于 {good:.1f}"

    def test_batch_no_inferred_from_date(self, fake_mat):
        with MatrBatch(fake_mat) as batch:
            assert batch.batch_date == "2017-05-12"
            assert batch.batch_no == 1

    def test_unknown_date_needs_explicit_batch_no(self, tmp_path):
        path = tmp_path / "mystery.mat"
        write_fake_matr(path, date="1999-01-01")
        with pytest.raises(MatrError, match="批次号"):
            MatrBatch(path)
        with MatrBatch(path, batch_no=7) as batch:
            assert batch.cell(0).cell_id == "b7c0"

    def test_inconsistent_field_length_rejected(self, tmp_path):
        path = tmp_path / "short.mat"
        write_fake_matr(path, drop_cycles=1)
        with pytest.raises(MatrError, match="不一致"):
            MatrBatch(path)

    def test_curve_index_out_of_range(self, fake_mat):
        with MatrBatch(fake_mat) as batch:
            with pytest.raises(MatrError, match="越界"):
                batch.curve(0, 99)

    def test_unknown_curve_field(self, fake_mat):
        with MatrBatch(fake_mat) as batch:
            with pytest.raises(MatrError, match="未知曲线字段"):
                batch.curve(0, 0, "nope")

    def test_cell_index_out_of_range(self, fake_mat):
        with MatrBatch(fake_mat) as batch:
            with pytest.raises(MatrError, match="越界"):
                batch.cell(99)

    def test_missing_file(self, tmp_path):
        with pytest.raises(MatrError, match="文件不存在"):
            MatrBatch(tmp_path / "nope.mat")

    def test_not_a_matr_file(self, tmp_path):
        path = tmp_path / "other.mat"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("something", data=np.arange(3))
        with pytest.raises(MatrError, match="batch"):
            MatrBatch(path)

    def test_local_file_reports_no_requests(self, fake_mat):
        with MatrBatch(fake_mat) as batch:
            batch.summary(0)
            assert batch.requests == 0
            assert batch.bytes_fetched == 0
            assert batch.n_bytes == fake_mat.stat().st_size

    def test_mirror_url(self):
        assert mirror_url(1).endswith("2017-05-12_batchdata_updated_struct_errorcorrect.mat")
        with pytest.raises(MatrError):
            mirror_url(9)


# ---------------------------------------------------------------------------
# HTTPRangeFile：Range 语义
# ---------------------------------------------------------------------------
class _RangeHandler(http.server.BaseHTTPRequestHandler):
    payload = b""
    honor_range = True
    # HTTP/1.1 + Content-Length 才会让客户端复用连接；用默认的 1.0 每条
    # 请求都关连接，「连接复用」这件事在测试里根本观察不到。
    protocol_version = "HTTP/1.1"
    connections: set = set()

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler 的约定
        type(self).connections.add(self.client_address)
        spec = self.headers.get("Range")
        if self.honor_range and spec and spec.startswith("bytes="):
            start_s, _, end_s = spec[len("bytes=") :].partition("-")
            start = int(start_s)
            end = int(end_s) if end_s else len(self.payload) - 1
            body = self.payload[start : end + 1]
            self.send_response(206)
            self.send_header(
                "Content-Range",
                f"bytes {start}-{start + len(body) - 1}/{len(self.payload)}",
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *args):  # 静音测试输出
        return None


def _serve(payload: bytes, *, honor_range: bool = True):
    handler = type(
        "Handler", (_RangeHandler,), {"payload": payload, "honor_range": honor_range}
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/blob"


def _serve_counting(payload: bytes):
    """额外返回一个记录 TCP 连接（客户端端口）的集合，用来验证连接复用。"""
    handler = type(
        "CountingHandler",
        (_RangeHandler,),
        {"payload": payload, "honor_range": True, "connections": set()},
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/blob", handler.connections


@pytest.fixture
def payload():
    return bytes(range(256)) * 40  # 10 KB，内容可预测


@pytest.fixture
def server(payload):
    server, url = _serve(payload)
    yield url
    server.shutdown()
    server.server_close()


class TestHTTPRangeFile:
    def test_size_from_content_range(self, server, payload):
        handle = HTTPRangeFile(server, retry_delay=0.01)
        assert handle.size == len(payload)

    def test_sequential_reads(self, server, payload):
        chunks = []
        with HTTPRangeFile(server, retry_delay=0.01) as handle:
            while True:
                chunk = handle.read(1000)
                if not chunk:
                    break
                chunks.append(chunk)
        assert b"".join(chunks) == payload

    def test_seek_and_read(self, server, payload):
        with HTTPRangeFile(server, retry_delay=0.01) as handle:
            handle.seek(5000)
            assert handle.tell() == 5000
            assert handle.read(100) == payload[5000:5100]
            handle.seek(-100, 2)
            assert handle.read() == payload[-100:]

    def test_prefetch_then_read_is_cached(self, server, payload):
        with HTTPRangeFile(server, retry_delay=0.01) as handle:
            handle.prefetch([(0, len(payload))])
            before = handle.requests
            assert handle.read(64) == payload[:64]
            handle.seek(len(payload) - 32)
            assert handle.read(32) == payload[-32:]
            assert handle.requests == before, "预取过的区间不应再发请求"

    def test_short_read_at_eof(self, server, payload):
        with HTTPRangeFile(server, retry_delay=0.01) as handle:
            handle.seek(len(payload) - 10)
            assert handle.read(999) == payload[-10:]
            assert handle.read(1) == b""

    def test_second_batch_opens_no_new_connection(self):
        """两批请求之间不能重新握手。

        这条挡的是一个**我自己写进去又量错过的** bug：``_pull`` 每次新建
        ``ThreadPoolExecutor``，``with`` 退出时把线程 join 掉，线程本地的
        连接随之失效，于是每一批请求都要重做一次 TLS 握手。表现为
        「并发看起来有害」——我据此把 ``workers`` 从 8 改成 1，还写了
        「并发有害，别改回去」。

        真正的因果是连接复用：同主机同时段实测，新建连接 0.63 请求/秒、
        复用连接 2.56、复用连接 + 8 线程 8.45。所以钉住「第二批不开新连接」
        比钉住某个并发数字有用——数字可以再调，这条不能破。
        """
        payload = bytes(range(256)) * 40
        server, url, conns = _serve_counting(payload)
        try:
            with HTTPRangeFile(url, workers=2, retry_delay=0.01) as handle:
                # 第一批把线程池和连接都建起来
                handle.prefetch([(0, 100), (500, 600)])
                assert handle.requests == 2
                warm = len(conns)
                assert warm >= 1, "第一批居然没建连接"
                # 第二批：走同一个池、同一批线程、同一批连接
                handle.prefetch([(1000, 1100), (1500, 1600)])
                assert handle.requests == 4
                assert len(conns) == warm, (
                    f"第二批新开了 {len(conns) - warm} 条连接；"
                    "线程池或线程本地连接没有跨 _pull 存活"
                )
        finally:
            server.shutdown()
            server.server_close()

    def test_close_shuts_down_pool_and_connections(self):
        """``close()`` 之后不能留下活线程——否则进程退出会挂住。"""
        payload = bytes(range(256)) * 40
        server, url, _ = _serve_counting(payload)
        try:
            handle = HTTPRangeFile(url, workers=4, retry_delay=0.01)
            handle.prefetch([(0, 100), (200, 300), (400, 500), (600, 700)])
            assert handle._pool is not None, "多段预取应该建起线程池"
            handle.close()
            assert handle._pool is None
            assert handle._conns == []
        finally:
            server.shutdown()
            server.server_close()

    def test_default_workers(self):
        """默认 8：并发有 5.5 倍收益，前提是连接复用（见类 docstring）。

        从签名读默认值而**不构造实例**：构造函数会探测远端文件大小，
        那是一条真实请求，测试不该为它付钱。
        """
        import inspect

        default = inspect.signature(HTTPRangeFile.__init__).parameters["workers"].default
        assert default == 8, "内部补洞并发默认值被改小了；见 docstring 里的实测表"

    def test_rejects_server_ignoring_range(self, payload):
        server, url = _serve(payload, honor_range=False)
        try:
            with pytest.raises(MatrError):
                HTTPRangeFile(url, retry_delay=0.01)
        finally:
            server.shutdown()
            server.server_close()

    def test_http_scheme_rejected(self):
        with pytest.raises(MatrError, match="http"):
            HTTPRangeFile("ftp://example.com/blob")


# ---------------------------------------------------------------------------
# 延续关系的证据逻辑（纯函数，不联网）


class TestContinuationEvidence:
    """``build_manifest`` 的四条证据里有三条是纯逻辑，这里逐条钉住。

    为什么要钉：这套判据换过一次。第一版用 ``channel_id`` 判断「同一颗电池」，
    实测 5 对全 FAIL——因为 ``channel_id`` 的第 5 位是批次内行号，跨批次必然
    不同。判据错了的表现是「验证失败」，而验证失败的默认反应是「那就别删」，
    于是 5 颗重复电池会静默留在数据集里，指标虚高而曲线看不出来。
    所以每条证据都要有正控和负控。
    """

    @staticmethod
    def _manifest():
        from build_manifest import CONTINUATION_ADD_LEN, CONTINUATIONS

        # 用真实 policy 字符串：残缺的后缀是数据里的事实，不是笔误。
        origin_policy = {
            "b1c0": "3.6C(80%)-3.6C",
            "b1c1": "3.6C(80%)-3.6C",
            "b1c2": "3.6C(80%)-3.6C",
            "b1c3": "4C(80%)-4C",
            "b1c4": "4C(80%)-4C",
            "b1c5": "4.4C(80%)-4.4C",
            "b1c6": "5.4C(80%)-5.4C",
        }
        cont_policy = {
            "b2c7": "80%)-3.6C",
            "b2c8": "80%)-3.6C",
            "b2c9": "80%)-3.6C",
            "b2c15": "80%)-4C",
            "b2c16": "80%)-4C",
        }
        cells1 = [
            {
                "cell_id": cid,
                "index": int(cid.split("c")[1]),
                "policy": policy,
                "n_cycles": 1000,
            }
            for cid, policy in origin_policy.items()
        ]
        cells2 = [
            {
                "cell_id": cid,
                "index": int(cid.split("c")[1]),
                "policy": policy,
                "n_cycles": CONTINUATION_ADD_LEN[cid] + 1,
            }
            for cid, policy in cont_policy.items()
        ]
        assert set(cont_policy) == set(CONTINUATIONS)
        return {
            "batches": [
                {"batch_no": 1, "cells_meta_complete": True, "cells": cells1},
                {"batch_no": 2, "cells_meta_complete": True, "cells": cells2},
            ]
        }

    def test_suffix_match_accepts_real_truncation(self):
        from build_manifest import policy_suffix_match

        assert policy_suffix_match("80%)-3.6C", "3.6C(80%)-3.6C")
        assert policy_suffix_match("80%)-4C", "4C(80%)-4C")

    def test_suffix_match_rejects_wrong_group(self):
        from build_manifest import policy_suffix_match

        # 4.4C 那条的后缀是 80%)-4.4C，接不上 80%)-4C——这正是后缀能定组的原因
        assert not policy_suffix_match("80%)-4C", "4.4C(80%)-4.4C")
        assert not policy_suffix_match("80%)-3.6C", "4C(80%)-4C")
        assert not policy_suffix_match("", "4C(80%)-4C")

    def test_policy_candidates_pin_the_group(self):
        from build_manifest import policy_candidates

        policies = {c["cell_id"]: c["policy"] for c in self._manifest()["batches"][0]["cells"]}
        assert policy_candidates("80%)-3.6C", policies) == ["b1c0", "b1c1", "b1c2"]
        assert policy_candidates("80%)-4C", policies) == ["b1c3", "b1c4"]

    def test_evidence_passes_on_real_shaped_manifest(self):
        from build_manifest import check_continuation_evidence

        assert check_continuation_evidence(self._manifest()) == []

    def test_evidence_unverifiable_without_meta(self):
        from build_manifest import check_continuation_evidence

        thin = self._manifest()
        thin["batches"][1]["cells_meta_complete"] = False
        assert check_continuation_evidence(thin) is None, "没有元信息时必须返回 None，不能假装通过"

    def test_evidence_catches_wrong_policy_group(self):
        """负控：把 b2c15 指向 4.4C 那颗，后缀就接不上。"""
        from build_manifest import CONTINUATIONS, check_continuation_evidence

        bad = self._manifest()
        for cell in bad["batches"][1]["cells"]:
            if cell["cell_id"] == "b2c15":
                cell["policy"] = "80%)-4.4C"
        problems = check_continuation_evidence(bad)
        assert problems, "policy 接不上却没报错"

    def test_evidence_catches_wrong_segment_length(self):
        """负控：段长对不上 add_len，说明「谁接谁」配错了。"""
        from build_manifest import check_continuation_evidence

        bad = self._manifest()
        for cell in bad["batches"][1]["cells"]:
            if cell["cell_id"] == "b2c7":
                cell["n_cycles"] = 980  # b2c8 的长度
        problems = check_continuation_evidence(bad)
        assert any("循环" in p for p in problems), problems

    def test_evidence_catches_double_booking(self, monkeypatch):
        """负控：两颗延续段抢同一颗原段，必须报出来。

        这一条只能靠 monkeypatch 造——正常数据里映射是一对一的。但它是真会发生的
        错误：抄映射时多写一行，两颗延续段就都指向同一颗电池，于是一颗电池被
        扣了两次，135 变成 134，而所有检查都还是 PASS。
        """
        import build_manifest
        from build_manifest import check_continuation_evidence

        doubled = dict(build_manifest.CONTINUATIONS)
        doubled["b2c8"] = "b1c0"  # b1c0 同时被 b2c7 和 b2c8 认领
        monkeypatch.setattr(build_manifest, "CONTINUATIONS", doubled)

        problems = check_continuation_evidence(self._manifest())
        assert any("同时认作原段" in p for p in problems), problems
