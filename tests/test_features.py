"""``build_features.py`` 的测试：断点续跑逻辑 + 用合成 HDF5 跑通抽取。

为什么值得测
------------
这个脚本以前是「跑完才算」：135 颗一次性抽完，中断就全部作废，而远端读一次
要 1.4 s 到几十秒，**它不可能一次跑完**。加上分片续跑之后，新的失效方式变成了：

1. 分片写一半被 kill，留下半块 parquet——它恰好还能被打开，于是下一次被当成
   有效数据跳过，特征表少了几百行而没有任何报错；
2. 换了电压网格接着跑，新旧行的第 i 列不是同一个物理量，拼出来的表看不出异常；
3. 并发写分片时行序不稳定，两次运行结果不可复现。

三条都只在结果上体现为「数字有点怪」，所以都要有测试钉住。
"""

from __future__ import annotations

import json

import pytest

# importorskip 必须在第三方 import 之前，理由同 test_matr.py。
h5py = pytest.importorskip("h5py", reason="matr 需要 h5py")
np = pytest.importorskip("numpy", reason="matr 需要 numpy")
pd = pytest.importorskip("pandas", reason="特征表需要 pandas")
pytest.importorskip("pyarrow", reason="parquet 需要 pyarrow")

import build_features as bf  # noqa: E402
from build_features import (  # noqa: E402
    cell_frame,
    concat_parts,
    extract_batch,
    parse_cycles,
    parts_dir_for,
    read_cell_part,
    state_fingerprint,
    state_mismatch,
    write_cell_part,
)
from build_manifest import resolve_source  # noqa: E402
# 同样从 flsoh.features 导入：flsoh.data 在模块层 import torch。
from flsoh.features import (  # noqa: E402
    CELL_COL,
    CYCLE_COL,
    FEATURE_SPEC_VERSION,
    SOH_COL,
    make_v_grid,
)
from flsoh.matr import MatrBatch  # noqa: E402

from test_matr import write_fake_matr  # noqa: E402


# ---------------------------------------------------------------------------
# 规格指纹


class TestStateFingerprint:
    def test_identical_state_matches(self):
        state = state_fingerprint(
            cycles=[5, 10], ref_cycle=1, v_lo=2.0, v_hi=3.5, points=1000
        )
        assert state_mismatch(dict(state), state) is None

    def test_no_saved_state_matches(self):
        """第一次跑，没有 state.json，不能当成不匹配。"""
        assert state_mismatch(None, {"points": 1000}) is None

    def test_changed_grid_is_refused(self):
        saved = state_fingerprint(
            cycles=[5, 10], ref_cycle=1, v_lo=2.0, v_hi=3.5, points=1000
        )
        other = state_fingerprint(
            cycles=[5, 10], ref_cycle=1, v_lo=2.0, v_hi=3.5, points=500
        )
        reason = state_mismatch(saved, other)
        assert reason and "points" in reason and "--restart" in reason

    def test_changed_cycles_is_refused(self):
        saved = state_fingerprint(
            cycles=[5, 10, 15], ref_cycle=1, v_lo=2.0, v_hi=3.5, points=1000
        )
        other = state_fingerprint(
            cycles=[5, 10], ref_cycle=1, v_lo=2.0, v_hi=3.5, points=1000
        )
        assert state_mismatch(saved, other) is not None

    def test_list_vs_tuple_does_not_matter(self):
        """JSON 读回来是 list；若哪天内部改成 tuple，不该因为类型差异拒绝续跑。"""
        saved = {"cycles": [5, 10], "points": 1000}
        assert state_mismatch(saved, {"cycles": (5, 10), "points": 1000}) is None

    def test_fingerprint_carries_the_algorithm_version(self):
        state = state_fingerprint(
            cycles=[5, 10], ref_cycle=1, v_lo=2.0, v_hi=3.5, points=1000
        )
        assert state["feature_version"] == FEATURE_SPEC_VERSION

    def test_parts_without_version_are_refused(self):
        """负控：2026-10-25 之前的 state.json 没有这个键。

        那批分片是用「整条循环插值」算的，而参数一个都没变。少了这道比对，
        重跑同一条命令会把它们当成有效数据全部跳过——表里一行新数据都没有，
        脚本还会打印「全部有分片」并退出 0。
        """
        kwargs = dict(cycles=[5, 10], ref_cycle=1, v_lo=2.0, v_hi=3.5, points=1000)
        saved = state_fingerprint(**kwargs)
        del saved["feature_version"]
        reason = state_mismatch(saved, state_fingerprint(**kwargs))
        assert reason and "feature_version" in reason and "--restart" in reason

    def test_changed_algorithm_version_is_refused(self, monkeypatch):
        kwargs = dict(cycles=[5, 10], ref_cycle=1, v_lo=2.0, v_hi=3.5, points=1000)
        saved = state_fingerprint(**kwargs)
        monkeypatch.setattr(bf, "FEATURE_SPEC_VERSION", "discharge-v99")
        reason = state_mismatch(saved, state_fingerprint(**kwargs))
        assert reason and "feature_version" in reason


# ---------------------------------------------------------------------------
# 分片


def _sample_frame(cell_id: str, cycles=(5, 10), n_points: int = 8) -> "pd.DataFrame":
    rows = np.arange(len(cycles) * n_points, dtype=np.float32).reshape(
        len(cycles), n_points
    )
    frame = pd.DataFrame(rows, columns=[f"f{i}" for i in range(n_points)])
    frame.insert(0, CELL_COL, [cell_id] * len(cycles))
    frame.insert(1, CYCLE_COL, np.asarray(cycles, dtype=np.int32))
    frame.insert(2, SOH_COL, np.linspace(1.0, 0.9, len(cycles), dtype=np.float32))
    return frame


class TestCellParts:
    def test_round_trip(self, tmp_path):
        parts = parts_dir_for(tmp_path / "features.parquet")
        path = write_cell_part(parts, "b1c0", _sample_frame("b1c0"))
        assert path == parts / "b1c0.parquet"
        assert not list(parts.glob("*.tmp")), "临时文件必须被换掉，不能留下"
        back = read_cell_part(path, cell_id="b1c0", n_points=8, n_cycles=2)
        assert back is not None and len(back) == 2

    def test_rejects_wrong_grid(self, tmp_path):
        """负控：网格点数不同的分片必须被拒。

        这不是洁癖。换成 500 点接着跑时，旧分片的第 i 列和新分片的第 i 列
        不是同一个电压，拼起来不报任何异常。
        """
        parts = parts_dir_for(tmp_path / "features.parquet")
        path = write_cell_part(parts, "b1c0", _sample_frame("b1c0"))
        assert read_cell_part(path, cell_id="b1c0", n_points=500, n_cycles=2) is None

    def test_rejects_wrong_cell_id(self, tmp_path):
        parts = parts_dir_for(tmp_path / "features.parquet")
        path = write_cell_part(parts, "b1c0", _sample_frame("b1c0"))
        assert read_cell_part(path, cell_id="b1c1", n_points=8, n_cycles=2) is None

    def test_rejects_duplicated_cycles(self, tmp_path):
        parts = parts_dir_for(tmp_path / "features.parquet")
        path = write_cell_part(parts, "b1c0", _sample_frame("b1c0", cycles=(5, 5)))
        assert read_cell_part(path, cell_id="b1c0", n_points=8, n_cycles=2) is None

    def test_rejects_wrong_row_count(self, tmp_path):
        parts = parts_dir_for(tmp_path / "features.parquet")
        path = write_cell_part(parts, "b1c0", _sample_frame("b1c0"))
        assert read_cell_part(path, cell_id="b1c0", n_points=8, n_cycles=3) is None

    def test_rejects_truncated_file(self, tmp_path):
        """负控：写一半被 kill 留下的残片必须被拒。

        parquet 有 footer，截断的文件通常打不开；但「通常」不够，
        必须有一个显式判据，并且要验证它在打不开时返回 None 而不是抛异常。
        """
        parts = parts_dir_for(tmp_path / "features.parquet")
        path = write_cell_part(parts, "b1c0", _sample_frame("b1c0"))
        path.write_bytes(path.read_bytes()[:20])
        assert read_cell_part(path, cell_id="b1c0", n_points=8, n_cycles=2) is None

    def test_rejects_garbage(self, tmp_path):
        parts = parts_dir_for(tmp_path / "features.parquet")
        path = write_cell_part(parts, "b1c0", _sample_frame("b1c0"))
        path.write_bytes(b"not a parquet file at all")
        assert read_cell_part(path, cell_id="b1c0", n_points=8, n_cycles=2) is None


class TestConcatParts:
    def test_sorted_deterministically(self, tmp_path):
        parts = parts_dir_for(tmp_path / "features.parquet")
        write_cell_part(parts, "b1c2", _sample_frame("b1c2", cycles=(5, 10)))
        write_cell_part(parts, "b1c0", _sample_frame("b1c0", cycles=(5, 10)))
        write_cell_part(parts, "b1c1", _sample_frame("b1c1", cycles=(5, 10)))

        forward = concat_parts(sorted(parts.glob("*.parquet")), 8)
        backward = concat_parts(sorted(parts.glob("*.parquet"), reverse=True), 8)
        assert list(forward[CELL_COL]) == ["b1c0", "b1c0", "b1c1", "b1c1", "b1c2", "b1c2"]
        assert list(forward[CELL_COL]) == list(backward[CELL_COL])
        assert list(forward[CYCLE_COL]) == list(backward[CYCLE_COL])

    def test_cycle_order_within_cell(self, tmp_path):
        parts = parts_dir_for(tmp_path / "features.parquet")
        write_cell_part(parts, "b1c0", _sample_frame("b1c0", cycles=(10, 5)))
        frame = concat_parts(sorted(parts.glob("*.parquet")), 8)
        assert list(frame[CYCLE_COL]) == [5, 10]


class TestCellFrame:
    def test_none_when_no_rows(self):
        from build_features import CellResult

        assert cell_frame(CellResult(cell_id="b1c0"), 8) is None

    def test_columns_are_cell_cycle_soh_then_features(self):
        from build_features import CellResult

        frame = cell_frame(
            CellResult(
                cell_id="b1c0",
                cycles=[5],
                soh=[1.0],
                rows=[np.zeros(8, dtype=np.float32)],
            ),
            8,
        )
        assert list(frame.columns) == [CELL_COL, CYCLE_COL, SOH_COL] + [
            f"f{i}" for i in range(8)
        ]


class TestParseCycles:
    def test_default_range(self):
        assert parse_cycles("5:100:5") == list(range(5, 101, 5))

    def test_rejects_bad_spec(self):
        for bad in ("5:100", "0:100:5", "100:5:5", "5:100:0"):
            with pytest.raises(SystemExit):
                parse_cycles(bad)


# ---------------------------------------------------------------------------
# 端到端：合成 HDF5 上跑抽取，全程不联网


class TestExtractAgainstFakeMatr:
    def test_extracts_expected_rows(self, tmp_path):
        path = tmp_path / "2017-05-12_fake.mat"
        write_fake_matr(path, n_cells=3, n_cycles=8, n_points=32)
        entry = {
            "batch_no": 1,
            "cells": [
                {"cell_id": f"b1c{i}", "index": i} for i in range(3)
            ],
        }
        # 合成曲线的电压区间是 [3.0, 3.6]，网格必须落在里面，否则会整条丢
        grid = make_v_grid(3.0, 3.5, 50)
        results = extract_batch(
            entry,
            str(path),
            cycles=[1, 3, 5],
            ref_cycle=1,
            grid=grid,
            workers=1,
        )
        assert [r.cell_id for r in results] == ["b1c0", "b1c1", "b1c2"]
        for result in results:
            assert result.cycles == [1, 3, 5]
            assert not result.drops, result.drops
            for row in result.rows:
                assert row.shape == (50,)

    def test_soh_is_capacity_ratio(self, tmp_path):
        """标签必须是 QD_k / QD_ref，不是别的归一化方式。"""
        path = tmp_path / "2017-05-12_fake.mat"
        write_fake_matr(path, n_cells=1, n_cycles=8, n_points=32)
        with MatrBatch(str(path), batch_no=1) as batch:
            qd = batch.summary(0, fields=("QDischarge",))["QDischarge"]
        grid = make_v_grid(3.0, 3.5, 50)
        entry = {"batch_no": 1, "cells": [{"cell_id": "b1c0", "index": 0}]}
        results = extract_batch(
            entry, str(path), cycles=[1, 3, 5], ref_cycle=1, grid=grid, workers=1
        )
        expected = [float(qd[k]) / float(qd[1]) for k in (1, 3, 5)]
        assert results[0].soh == pytest.approx(expected)

    def test_out_of_range_cycles_are_dropped_not_raised(self, tmp_path):
        """电池只有 8 个循环却要第 100 个：记 drop，不能抛异常中断整批。"""
        path = tmp_path / "2017-05-12_fake.mat"
        write_fake_matr(path, n_cells=1, n_cycles=8, n_points=32)
        grid = make_v_grid(3.0, 3.5, 50)
        entry = {"batch_no": 1, "cells": [{"cell_id": "b1c0", "index": 0}]}
        results = extract_batch(
            entry, str(path), cycles=[1, 100], ref_cycle=1, grid=grid, workers=1
        )
        assert results[0].cycles == [1]
        assert any("个循环" in d["reason"] for d in results[0].drops)

    def test_workers_greater_than_cells(self, tmp_path):
        """workers 比电池多时不能崩，也不能少抽。"""
        path = tmp_path / "2017-05-12_fake.mat"
        write_fake_matr(path, n_cells=2, n_cycles=8, n_points=32)
        grid = make_v_grid(3.0, 3.5, 50)
        entry = {
            "batch_no": 1,
            "cells": [{"cell_id": f"b1c{i}", "index": i} for i in range(2)],
        }
        results = extract_batch(
            entry, str(path), cycles=[1, 3], ref_cycle=1, grid=grid, workers=8
        )
        assert len(results) == 2
        assert all(len(r.rows) == 2 for r in results)

    def test_resolves_local_source_over_mirror(self, tmp_path):
        """本地有 .mat 就必须用本地，不能偷偷联网。"""
        raw = tmp_path / "raw"
        raw.mkdir()
        write_fake_matr(
            raw / "2017-05-12_batchdata_updated_struct_errorcorrect.mat", n_cells=1
        )
        assert resolve_source(1, raw).endswith(".mat")
        assert "github" not in resolve_source(1, raw)


# ---------------------------------------------------------------------------
# 端到端：main() 的续跑路径


class TestResumeEndToEnd:
    """``main()`` 的断点续跑路径。

    分片逻辑单测过了还不够——**续跑这条路径本身**要真的跑一遍。
    「写分片」和「跳过已有分片」是两段代码，中间那段从没被执行过的话，
    第一次真跑就会在 135 颗的第十颗上炸掉。
    """

    @staticmethod
    def _setup(tmp_path, *, n_cells=3):
        raw = tmp_path / "raw"
        raw.mkdir()
        from flsoh.matr import BATCH_FILES

        write_fake_matr(raw / BATCH_FILES[1], n_cells=n_cells, n_cycles=8)
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
    def _args(tmp_path, manifest, raw, out, extra=()):
        return [
            "--manifest", str(manifest),
            "--out", str(out),
            "--source-dir", str(raw),
            "--cycles", "1:7:2",
            "--v-lo", "3.0",
            "--v-hi", "3.5",
            "--points", "50",
            *extra,
        ]

    def test_first_run_writes_table_and_spec(self, tmp_path):
        from build_features import main

        manifest, raw = self._setup(tmp_path)
        out = tmp_path / "features.parquet"
        assert main(self._args(tmp_path, manifest, raw, out)) == 0

        assert out.is_file()
        spec = json.loads(
            (tmp_path / "features.spec.json").read_text(encoding="utf-8")
        )
        assert spec["rows"] == 12, "3 颗 × 4 个循环"
        assert spec["cells_with_rows"] == 3
        assert spec["dropped"]["by_reason"] == {}
        assert not parts_dir_for(out).is_dir(), "跑完默认要清掉分片目录"

    def test_second_run_reuses_parts(self, tmp_path, capsys):
        from build_features import main

        manifest, raw = self._setup(tmp_path)
        out = tmp_path / "features.parquet"
        # 第一次要 --keep-parts：跑完默认清掉分片目录，清掉之后本来就没得续。
        assert main(self._args(tmp_path, manifest, raw, out, ("--keep-parts",))) == 0
        first = out.read_bytes()

        capsys.readouterr()
        assert main(self._args(tmp_path, manifest, raw, out)) == 0
        printed = capsys.readouterr().out
        assert "全部有分片" in printed, "已有分片时必须跳过，不能重抽"
        assert out.read_bytes() == first, "同样的输入必须产出同样的字节"

    def test_changed_grid_refuses_to_resume(self, tmp_path):
        from build_features import main

        manifest, raw = self._setup(tmp_path)
        out = tmp_path / "features.parquet"
        assert main(self._args(tmp_path, manifest, raw, out, ("--keep-parts",))) == 0

        args = self._args(tmp_path, manifest, raw, out, ("--points", "25"))
        with pytest.raises(SystemExit) as excinfo:
            main(args)
        assert "--restart" in str(excinfo.value)
        assert out.read_bytes(), "被拒绝之后不能把旧表删掉"

    def test_restart_clears_parts(self, tmp_path):
        from build_features import main

        manifest, raw = self._setup(tmp_path)
        out = tmp_path / "features.parquet"
        assert main(self._args(tmp_path, manifest, raw, out)) == 0
        assert main(self._args(tmp_path, manifest, raw, out, ("--restart",))) == 0
        spec = json.loads(
            (tmp_path / "features.spec.json").read_text(encoding="utf-8")
        )
        assert spec["rows"] == 12

    def test_keep_parts_keeps_directory(self, tmp_path):
        from build_features import main

        manifest, raw = self._setup(tmp_path)
        out = tmp_path / "features.parquet"
        main(self._args(tmp_path, manifest, raw, out, ("--keep-parts",)))
        parts = parts_dir_for(out)
        assert parts.is_dir()
        assert len(list(parts.glob("*.parquet"))) == 3
        assert (parts / "state.json").is_file()

    def test_resume_after_losing_one_part(self, tmp_path):
        """负控：删掉一块分片后重跑，必须把那颗补回来，行数恢复。"""
        from build_features import main

        manifest, raw = self._setup(tmp_path)
        out = tmp_path / "features.parquet"
        main(self._args(tmp_path, manifest, raw, out, ("--keep-parts",)))
        (parts_dir_for(out) / "b1c1.parquet").unlink()

        assert main(self._args(tmp_path, manifest, raw, out)) == 0
        frame = pd.read_parquet(out)
        assert len(frame) == 12
        assert set(frame[CELL_COL]) == {"b1c0", "b1c1", "b1c2"}

    def test_limit_cells_then_full_run(self, tmp_path):
        """先跑前 1 颗，再全量：第二次必须跳过第一颗，把剩下的补齐。"""
        from build_features import main

        manifest, raw = self._setup(tmp_path, n_cells=3)
        out = tmp_path / "features.parquet"
        main(self._args(tmp_path, manifest, raw, out, ("--limit-cells", "1")))
        assert len(pd.read_parquet(out)) == 4

        main(self._args(tmp_path, manifest, raw, out))
        frame = pd.read_parquet(out)
        assert len(frame) == 12

    def test_completed_run_clears_parts_so_next_run_starts_fresh(self, tmp_path):
        """跑完之后分片目录默认清掉，所以再跑一次是全新一轮，不是续跑。

        这条语义要钉住，否则会有人以为「同一条命令重复跑会自动跳过」。
        它只在**上一次没跑完**（或显式 --keep-parts）时才跳过。
        """
        from build_features import main

        manifest, raw = self._setup(tmp_path)
        out = tmp_path / "features.parquet"
        main(self._args(tmp_path, manifest, raw, out))
        assert not parts_dir_for(out).is_dir()

        assert main(self._args(tmp_path, manifest, raw, out)) == 0
        assert len(pd.read_parquet(out)) == 12, "重新抽一遍，结果必须一样"

