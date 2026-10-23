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
            v_refs, q_refs = [], []
            for k in range(n_cycles):
                voltage = np.linspace(3.0, 3.6, n_points)
                capacity = np.linspace(0.0, float(base[k]), n_points)
                v_refs.append(h5.create_dataset(f"/fake/c{i}/v{k}", data=voltage).ref)
                q_refs.append(h5.create_dataset(f"/fake/c{i}/q{k}", data=capacity).ref)
            cycles.create_dataset("V", data=_ref_column(v_refs))
            cycles.create_dataset("Qc", data=_ref_column(q_refs))
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
        with MatrBatch(fake_mat) as batch:
            voltage, capacity = batch.charge_curve(2, 3)
            assert voltage.shape == (32,)
            assert capacity.shape == (32,)
            assert voltage[0] == pytest.approx(3.0)
            assert voltage[-1] == pytest.approx(3.6)

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

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler 的约定
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

    def test_default_workers_is_one(self):
        """内部补洞并发默认必须是 1。

        这不是性能洁癖，是一个被实测推翻过的默认值：最早按「单连接 17 KB/s、
        6 并发 131 KB/s」设成 8，后来这个镜像改成**惩罚并发连接**，
        于是 8 从优化变成了 3 倍以上的减速——而顶层 ``--workers`` 调到 1
        也看不出问题，因为并发藏在 ``_pull`` 里面。

        同一个时段、同一份代码实测：内部并发 1 → 134 请求 / 436 KB / 60.6 s；
        内部并发 8 → 280 s 内没跑完。所以「并发听起来更快」在这里是错的，
        拿这条测试挡着，别让人凭直觉改回去。

        从签名读默认值而**不构造实例**：构造函数会探测远端文件大小，
        那是一条真实请求，测试不该为它付钱。
        """
        import inspect

        default = inspect.signature(HTTPRangeFile.__init__).parameters["workers"].default
        assert default == 1, "内部补洞并发默认值被改大了；见 docstring 里的实测表"

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
