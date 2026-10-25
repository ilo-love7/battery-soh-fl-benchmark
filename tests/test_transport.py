"""传输错误 ≠ 数据缺失：这一整类失败以前被静默吞掉。

真实的触发过程
--------------
用真实数据跑 ``build_features.py`` 时，沙箱代理的 DNS 抽了一次：
``getaddrinfo failed``。批次 1 的 46 颗于是全部变成「0 行、丢弃 1 条」，
每颗落下一个 ``.empty`` 分片。

**危害不在"永久跳过"**——``todo`` 只认 ``.parquet``，所以下次续跑这 46 颗
还是会被重抽，数据没丢。真正的危害是**审计撒谎**：

- ``features.spec.json`` 的 ``dropped`` 里躺着 46 条「参考曲线读不出」，
  读起来就是"这 46 颗电池没有数据"，而事实是网络断了一次；
- 更糟的形态是**这一轮没崩**：46 颗失败、89 颗成功，命令正常退出、
  表正常写出、spec 正常生成。下游拿到一张少三分之一样本的表，
  而唯一的告警是一段被误分类的丢弃原因。

根因是 ``MatrError`` 把两件相反的事塞进了同一个类：

- 数据问题（循环越界、容量非正、曲线覆盖不住窗口）→ 该记 ``dropped``，
  这颗算跑完，spec 里有据可查；
- 传输问题（DNS、连接重置、HTTP 5xx、服务器忽略 Range）→ **不是数据缺失**，
  记进 ``dropped`` 就等于把一次网络抖动写成一条数据缺陷。

修法是拆出 ``MatrTransportError``，并让 ``build_features`` 在遇到它时
落盘已抽到的分片、然后大声中止（退出码 1），重跑同一条命令即可续跑。
``.empty`` 的判定也跟着收紧：只有**不含传输原因**的标记才算"跑完"，
旧版本写下的那些会被删掉并重新排队。

这一文件钉住四件事：判据本身、``extract_cell`` 抛而不记、
``.empty`` 的自愈与保留、以及**数据问题不能被误判成传输错误**
（最后一条是负控，判反了会让一颗真没数据的电池每轮都被重抽，
而报告一直说"网络问题请重跑"——于是没人去查数据）。
"""

from __future__ import annotations

import json

import pytest

# importorskip 必须在第三方 import 之前，理由同 test_matr.py。
h5py = pytest.importorskip("h5py", reason="matr 需要 h5py")
np = pytest.importorskip("numpy", reason="matr 需要 numpy")
pd = pytest.importorskip("pandas", reason="特征表需要 pandas")
pytest.importorskip("pyarrow", reason="parquet 需要 pyarrow")

from build_features import (  # noqa: E402
    _looks_like_transport,
    extract_cell,
    main,
    parts_dir_for,
)
from flsoh.features import FEATURE_SPEC_VERSION, make_v_grid  # noqa: E402
from flsoh.matr import MatrError, MatrTransportError  # noqa: E402

from test_matr import write_fake_matr  # noqa: E402


class TestLooksLikeTransport:
    """判据本身要先钉住，否则它会在某个没预料到的真实错误消息上失灵。"""

    def test_real_dns_failure_is_transport(self):
        # 这就是逼出这一轮改动的原始错误消息，逐字抄下来
        reason = (
            "参考曲线读不出：区间抓取失败 [21760, 22272)："
            "gaierror: [Errno 11001] getaddrinfo failed"
        )
        assert _looks_like_transport(reason)

    def test_http_and_size_probe_are_transport(self):
        assert _looks_like_transport("区间抓取失败 [0, 512)：HTTP 502")
        assert _looks_like_transport("无法从 Content-Range 得知远端文件大小")

    def test_connection_errors_are_transport(self):
        assert _looks_like_transport("区间抓取失败：ConnectionResetError")
        assert _looks_like_transport("区间抓取失败：ConnectionRefusedError")
        assert _looks_like_transport("区间抓取失败：RemoteDisconnected")
        assert _looks_like_transport("区间抓取失败：timed out")

    def test_data_problems_are_not_transport(self):
        """负控：真正的数据问题必须**不**被判成传输错误。

        判反了的后果是静默的：一颗真的没有数据的电池会被反复重抽，
        每轮都失败，而报告一直说"网络问题请重跑"——于是没人去查数据。
        """
        assert not _looks_like_transport("该电池只有 3 个循环")
        assert not _looks_like_transport("参考循环容量非正：-1.0000")
        assert not _looks_like_transport("参考循环 1 越界（共 1 个）")
        assert not _looks_like_transport("曲线覆盖不住电压窗口 [2.0, 3.5] V")
        assert not _looks_like_transport("summary 读不出：字段不齐")


class TestTransportErrorIsNotADrop:
    """``extract_cell`` 遇到传输错误必须抛，不能记进 drops。"""

    def test_summary_transport_failure_raises(self):
        class FakeBatch:
            def summary(self, index, *, fields=()):
                raise MatrTransportError("区间抓取失败 [0, 512)：gaierror")

        with pytest.raises(MatrTransportError):
            extract_cell(
                FakeBatch(),
                "b1c0",
                0,
                cycles=[5, 10],
                ref_cycle=1,
                grid=make_v_grid(3.0, 3.5, 8),
            )

    def test_ref_curve_transport_failure_raises(self):
        """参考曲线读不出时最容易误判——它的消息长得就像"这颗电池没数据"。"""

        class FakeBatch:
            def summary(self, index, *, fields=()):
                return {"QDischarge": np.ones(200), "cycle": np.arange(200)}

            def discharge_curve(self, index, cycle):
                raise MatrTransportError("区间抓取失败：ConnectionResetError")

        with pytest.raises(MatrTransportError):
            extract_cell(
                FakeBatch(),
                "b1c0",
                0,
                cycles=[5, 10],
                ref_cycle=1,
                grid=make_v_grid(3.0, 3.5, 8),
            )

    def test_mid_cycle_transport_failure_raises(self):
        """抽到第 8 个循环时断网：已抽到的 7 行必须丢掉，不能留半颗。"""

        class FakeBatch:
            def summary(self, index, *, fields=()):
                return {"QDischarge": np.ones(200), "cycle": np.arange(200)}

            def discharge_curve(self, index, cycle):
                if cycle > 30:
                    raise MatrTransportError("区间抓取失败：timed out")
                return np.linspace(3.0, 3.5, 32), np.linspace(0.0, 1.0, 32)

        with pytest.raises(MatrTransportError):
            extract_cell(
                FakeBatch(),
                "b1c0",
                0,
                cycles=[5, 10, 15, 20, 25, 30, 35, 40],
                ref_cycle=1,
                grid=make_v_grid(3.0, 3.5, 8),
            )

    def test_data_problem_still_becomes_a_drop(self):
        """正控：数据问题照旧记 drops，不能把两类混在一起之后全都抛。"""

        class FakeBatch:
            def summary(self, index, *, fields=()):
                return {"QDischarge": np.ones(3), "cycle": np.arange(3)}

            def discharge_curve(self, index, cycle):
                raise MatrError("曲线覆盖不住电压窗口 [2.0, 3.5] V")

        result = extract_cell(
            FakeBatch(),
            "b1c0",
            0,
            cycles=[5, 10],
            ref_cycle=1,
            grid=make_v_grid(3.0, 3.5, 8),
        )
        assert result.drops and "覆盖不住" in result.drops[0]["reason"]
        assert result.rows == []


class TestStaleEmptyMarkersAreHealed:
    """旧版本写下的 .empty 要能被识别并删掉，否则那 46 颗永远补不回来。"""

    @staticmethod
    def _setup(tmp_path, *, n_cells=2):
        raw = tmp_path / "raw"
        raw.mkdir()
        write_fake_matr(
            raw / "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
            n_cells=n_cells,
            n_cycles=8,
        )
        manifest = {
            "batches": [
                {
                    "batch_no": 1,
                    "batch_date": "2017-05-12",
                    "n_cells": n_cells,
                    "cells": [
                        {"cell_id": f"b1c{i}", "index": i} for i in range(n_cells)
                    ],
                }
            ]
        }
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path, raw

    @staticmethod
    def _argv(manifest, raw, out):
        return [
            "--manifest", str(manifest),
            "--out", str(out),
            "--source-dir", str(raw),
            "--cycles", "1:7:2",
            "--v-lo", "3.0",
            "--v-hi", "3.5",
            "--points", "50",
            # 跑完默认清掉分片目录，这里要留着才能断言分片本身
            "--keep-parts",
        ]

    def test_stale_marker_is_deleted_and_cell_retried(self, tmp_path, capsys):
        manifest, raw = self._setup(tmp_path)
        out = tmp_path / "features.parquet"
        parts = parts_dir_for(out)
        parts.mkdir(parents=True)
        # 模拟旧版本：一次 DNS 失败留下的标记，内容逐字照抄实测输出
        (parts / "b1c0.empty").write_text(
            json.dumps(
                [
                    {
                        "cycle": None,
                        "reason": "参考曲线读不出：区间抓取失败 [21760, 22272)："
                        "gaierror: [Errno 11001] getaddrinfo failed",
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (parts / "state.json").write_text(
            json.dumps(
                {
                    # 指纹里的算法版本必须写全：缺了它新代码会判「算法变了」
                    # 并拒绝续跑，而这条测试要验的是**续跑**。
                    "feature_version": FEATURE_SPEC_VERSION,
                    "cycles": [1, 3, 5, 7],
                    "ref_cycle": 1,
                    "v_lo": 3.0,
                    "v_hi": 3.5,
                    "points": 50,
                }
            ),
            encoding="utf-8",
        )

        rc = main(self._argv(manifest, raw, out))
        printed = capsys.readouterr().out
        assert rc == 0, printed
        assert "网络失败留下的" in printed, "自愈必须说出来，不能默默删"
        assert not (parts / "b1c0.empty").exists()
        assert (parts / "b1c0.parquet").exists(), "这颗必须被重新抽回来"
        assert len(pd.read_parquet(out)) == 8, "两颗 × 4 个循环（1,3,5,7）"

    def test_genuine_empty_marker_is_kept(self, tmp_path):
        """负控：真的是数据没问题的 .empty 不能被删，否则每轮都白跑一遍。"""
        manifest, raw = self._setup(tmp_path, n_cells=1)
        out = tmp_path / "features.parquet"
        parts = parts_dir_for(out)
        parts.mkdir(parents=True)
        (parts / "b1c0.empty").write_text(
            json.dumps([{"cycle": 5, "reason": "该电池只有 3 个循环"}]),
            encoding="utf-8",
        )
        (parts / "state.json").write_text(
            json.dumps(
                {
                    # 指纹里的算法版本必须写全：缺了它新代码会判「算法变了」
                    # 并拒绝续跑，而这条测试要验的是**续跑**。
                    "feature_version": FEATURE_SPEC_VERSION,
                    "cycles": [1, 3, 5, 7],
                    "ref_cycle": 1,
                    "v_lo": 3.0,
                    "v_hi": 3.5,
                    "points": 50,
                }
            ),
            encoding="utf-8",
        )

        # 只有 1 颗且它是 .empty，于是一块 parquet 都没有 → 明确报错退出
        with pytest.raises(SystemExit):
            main(self._argv(manifest, raw, out))
        assert (parts / "b1c0.empty").exists(), "数据问题的标记必须留着"


class TestMatrTransportErrorHierarchy:
    """异常层级：新类必须是 MatrError 的子类，否则旧的 except 会漏放。"""

    def test_is_subclass(self):
        assert issubclass(MatrTransportError, MatrError)

    def test_can_be_caught_as_matrerror(self):
        with pytest.raises(MatrError):
            raise MatrTransportError("区间抓取失败：gaierror")


class TestPartsAreWrittenPerCell:
    """分片必须**逐颗**落盘，不能等整块跑完。

    这一轮要跑一小时，而中断随时会来。主循环原本等 ``future.result()``
    返回才写分片，于是一次中断白丢一整块（默认 8 颗 × 约 60 s）。
    改成 worker 每抽完一颗就回调 ``persist`` 之后，这条路径本身要被钉住——
    它和「写分片」是两段代码，中间那段没被执行过的话，第一次真跑就会炸。
    """

    @staticmethod
    def _fake_source(tmp_path, n_cells=4):
        raw = tmp_path / "raw"
        raw.mkdir()
        path = raw / "2017-05-12_batchdata_updated_struct_errorcorrect.mat"
        write_fake_matr(path, n_cells=n_cells, n_cycles=8)
        return str(path)

    def test_on_result_fires_once_per_cell(self, tmp_path):
        from build_features import _extract_chunk
        from flsoh.features import make_v_grid

        source = self._fake_source(tmp_path)
        metas = [{"cell_id": f"b1c{i}", "index": i} for i in range(4)]
        seen: list[str] = []

        results, error = _extract_chunk(
            source,
            1,
            metas,
            cycles=[1, 3],
            ref_cycle=1,
            grid=make_v_grid(3.0, 3.5, 16),
            on_result=lambda r: seen.append(r.cell_id),
        )
        assert error is None
        assert seen == ["b1c0", "b1c1", "b1c2", "b1c3"], seen
        assert [r.cell_id for r in results] == seen

    def test_on_result_fires_before_transport_error(self, tmp_path, monkeypatch):
        """断网那一刻，前面已抽到的颗也必须已经交出去了。"""

        from build_features import _extract_chunk
        from flsoh.features import make_v_grid
        from flsoh.matr import MatrTransportError

        source = self._fake_source(tmp_path)
        metas = [{"cell_id": f"b1c{i}", "index": i} for i in range(3)]
        seen: list[str] = []

        real_extract_one = _extract_chunk.__globals__["_extract_one"]
        calls = {"n": 0}

        def flaky(batch, meta, **kwargs):
            calls["n"] += 1
            if calls["n"] > 2:
                raise MatrTransportError("区间抓取失败：gaierror")
            return real_extract_one(batch, meta, **kwargs)

        monkeypatch.setattr("build_features._extract_one", flaky)
        results, error = _extract_chunk(
            source,
            1,
            metas,
            cycles=[1, 3],
            ref_cycle=1,
            grid=make_v_grid(3.0, 3.5, 16),
            on_result=lambda r: seen.append(r.cell_id),
        )
        assert isinstance(error, MatrTransportError)
        assert seen == ["b1c0", "b1c1"], "失败前抽到的两颗必须已经交出去"
        assert [r.cell_id for r in results] == seen
