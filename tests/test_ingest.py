"""``flsoh.ingest`` 与 ``run_e2e.py`` 纯函数部分的测试。

为什么值得测
------------
入库层的三个动作（校验 / 去重 / 组装）**做错了都不报错**：

- 不去重 → 重试上传被算成新数据，客户端样本量虚高，联邦加权平均的权重直接错；
- 去重太狠（按 session_id 而不是按内容）→ 同一 ID 的不同内容被丢掉，
  而那恰恰是「用户重装 App 后 ID 复用」的场景；
- 标签对不上行 → 每一行的标签来自别的会话，指标照样算得出来。

所以每条都要有断言，而且负控（「故意造坏」）必须能报出来。
"""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy", reason="ingest 需要 numpy")
pd = pytest.importorskip("pandas", reason="特征表需要 pandas")

from flsoh.features import make_v_grid  # noqa: E402
from flsoh.ingest import (  # noqa: E402
    IngestError,
    ingest_sessions,
    read_sessions_jsonl,
    sessions_to_frame,
    truth_from_json,
)
from flsoh.schema import session_fingerprint  # noqa: E402
from flsoh.simulate import simulate_battery, simulate_fleet  # noqa: E402

GRID = make_v_grid(2.0, 3.5, 200)


def _fleet(n_cells: int = 2, n_cycles: int = 4, seed: int = 0):
    sessions, truths = simulate_fleet(n_cells, n_cycles=n_cycles, seed=seed)
    truth = {t["session_id"]: t["soh"] for t in truths}
    return sessions, truth, truths


# ---------------------------------------------------------------------------
# 读文件


class TestReadJsonl:
    def test_reads_every_line(self, tmp_path):
        sessions, _, _ = _fleet()
        path = tmp_path / "s.jsonl"
        path.write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in sessions) + "\n",
            encoding="utf-8",
        )
        loaded = read_sessions_jsonl(path)
        assert len(loaded) == len(sessions)

    def test_blank_lines_are_skipped(self, tmp_path):
        sessions, _, _ = _fleet()
        path = tmp_path / "s.jsonl"
        path.write_text(
            json.dumps(sessions[0]) + "\n\n\n" + json.dumps(sessions[1]) + "\n",
            encoding="utf-8",
        )
        assert len(read_sessions_jsonl(path)) == 2

    def test_broken_line_reports_its_line_number(self, tmp_path):
        """坏行必须带行号报出来，不能跳过。

        跳过的坏行会让「入库了多少条」这个数字悄悄变小，而下游看到的是一个
        完全正常的表。
        """
        path = tmp_path / "s.jsonl"
        path.write_text('{"ok": 1}\n{not json}\n', encoding="utf-8")
        with pytest.raises(IngestError, match=":2"):
            read_sessions_jsonl(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(IngestError, match="不存在"):
            read_sessions_jsonl(tmp_path / "nope.jsonl")


# ---------------------------------------------------------------------------
# 校验与去重


class TestIngest:
    def test_accepts_valid_sessions(self):
        sessions, _, _ = _fleet()
        report = ingest_sessions(sessions)
        assert report["n_accepted"] == len(sessions)
        assert report["n_invalid"] == 0

    def test_rejects_invalid_and_keeps_going(self):
        """一条坏数据不能让整批作废——但必须计数并给出原因。"""
        sessions, _, _ = _fleet()
        broken = json.loads(json.dumps(sessions[0]))
        broken["session_id"] = "broken"
        broken["records"][0]["V"] = 99.0  # 超出锂电物理区间
        report = ingest_sessions([*sessions, broken])
        assert report["n_invalid"] == 1
        assert report["n_accepted"] == len(sessions)
        assert report["errors"][-1]["session_id"] == "broken"

    def test_duplicate_upload_is_dropped(self):
        """App 重试上传：内容逐字相同 → 只留一份。"""
        sessions, _, _ = _fleet()
        report = ingest_sessions([*sessions, sessions[0], sessions[0]])
        assert report["n_accepted"] == len(sessions)
        assert report["n_duplicate"] == 2

    def test_same_id_different_content_is_kept(self):
        """按**内容**去重而不是按 ID：重装 App 后 ID 可能复用而内容不同。

        按 ID 去重会把第二份当成重复丢掉，而那正是另一个真实的循环。
        """
        sessions, _, _ = _fleet()
        clone = json.loads(json.dumps(sessions[0]))
        clone["records"][1]["V"] += 0.01  # 内容变了，ID 没变
        assert session_fingerprint(clone) != session_fingerprint(sessions[0])
        report = ingest_sessions([sessions[0], clone])
        assert report["n_accepted"] == 2
        assert report["n_duplicate"] == 0

    def test_empty_input(self):
        report = ingest_sessions([])
        assert report["n_total"] == 0 and report["n_accepted"] == 0

    def test_error_list_is_truncated(self):
        """几千条错误全打出来会把真正的原因埋掉。"""
        from flsoh.ingest import MAX_ERRORS

        broken = [{"session_id": f"bad{i}"} for i in range(MAX_ERRORS + 10)]
        report = ingest_sessions(broken)
        assert report["n_invalid"] == MAX_ERRORS + 10
        assert len(report["errors"]) == MAX_ERRORS


# ---------------------------------------------------------------------------
# 组装特征表


class TestSessionsToFrame:
    def test_frame_shape_and_columns(self):
        sessions, truth, _ = _fleet(n_cells=2, n_cycles=4)
        frame, dropped = sessions_to_frame(sessions, v_grid=GRID, truth=truth)
        assert dropped == []
        # 每颗 4 个会话，参考会话不产生行
        assert len(frame) == 2 * 3
        assert list(frame.columns[:3]) == ["cell_id", "cycle", "soh"]
        assert frame.shape[1] == 3 + len(GRID)

    def test_cycle_column_is_a_time_ordinal(self):
        """App 路径没有循环号，只有时间序；这一列只用于按时间切 train/val。"""
        sessions, truth, _ = _fleet(n_cells=1, n_cycles=4)
        frame, _ = sessions_to_frame(sessions, v_grid=GRID, truth=truth)
        assert frame["cycle"].tolist() == [1, 2, 3]

    def test_labels_come_from_truth(self):
        sessions, truth, truths = _fleet(n_cells=1, n_cycles=3)
        frame, _ = sessions_to_frame(sessions, v_grid=GRID, truth=truth)
        # 参考会话（第一个）不产生行，所以第 1 行的标签是第 2 个会话的
        assert frame["soh"].tolist() == pytest.approx(
            [truths[1]["soh"], truths[2]["soh"]]
        )

    def test_missing_label_is_refused(self):
        """缺标签就拒绝：填 NaN 的行会把损失静默拉向 0。"""
        sessions, truth, _ = _fleet(n_cells=1, n_cycles=3)
        truth.pop(sessions[1]["session_id"])
        with pytest.raises(IngestError, match="没有真值标签"):
            sessions_to_frame(sessions, v_grid=GRID, truth=truth)

    def test_sessions_are_sorted_by_time_not_input_order(self):
        """输入顺序打乱后结果必须一致——上传顺序不该影响特征。"""
        sessions, truth, _ = _fleet(n_cells=1, n_cycles=4)
        a, _ = sessions_to_frame(sessions, v_grid=GRID, truth=truth)
        b, _ = sessions_to_frame(list(reversed(sessions)), v_grid=GRID, truth=truth)
        assert a["soh"].tolist() == pytest.approx(b["soh"].tolist())
        assert np.allclose(
            a[[c for c in a.columns if c.startswith("f")]].to_numpy(),
            b[[c for c in b.columns if c.startswith("f")]].to_numpy(),
        )

    def test_empty_sessions_raise(self):
        with pytest.raises(IngestError, match="没有可用的会话"):
            sessions_to_frame([], v_grid=GRID, truth={})

    def test_duplicated_sessions_do_not_duplicate_rows(self):
        """入库去重之后才组装，所以重复会话不会变成两行。"""
        sessions, truth, _ = _fleet(n_cells=1, n_cycles=3)
        report = ingest_sessions([*sessions, sessions[1]])
        frame, _ = sessions_to_frame(report["accepted"], v_grid=GRID, truth=truth)
        assert len(frame) == 2


class TestTruthFromJson:
    def test_reads_the_writer_format(self):
        _, _, truths = _fleet()
        payload = {"n_sessions": len(truths), "sessions": truths}
        mapping = truth_from_json(payload)
        assert len(mapping) == len(truths)
        assert mapping[truths[0]["session_id"]] == pytest.approx(truths[0]["soh"])

    def test_skips_entries_without_soh(self):
        mapping = truth_from_json([{"session_id": "a", "soh": 0.9}, {"session_id": "b"}])
        assert mapping == {"a": 0.9}


# ---------------------------------------------------------------------------
# run_e2e 的划分逻辑（纯函数，不需要 torch）


class TestE2ePartition:
    def test_round_robin_is_balanced(self):
        from run_e2e import round_robin

        groups = round_robin([f"c{i}" for i in range(9)], 4)
        sizes = sorted(len(v) for v in groups.values())
        assert sizes[-1] - sizes[0] <= 1
        assert sum(sizes) == 9

    def test_round_robin_covers_every_cell_once(self):
        from run_e2e import round_robin

        cells = [f"c{i}" for i in range(8)]
        flat = [c for group in round_robin(cells, 3).values() for c in group]
        assert sorted(flat) == sorted(cells)

    def test_too_many_clients_is_refused(self):
        from run_e2e import round_robin

        with pytest.raises(ValueError, match="分不出"):
            round_robin(["a", "b"], 3)

    def test_test_split_is_disjoint_and_deterministic(self):
        from run_e2e import split_test_cells

        train, test = split_test_cells([f"c{i}" for i in range(12)], 0.25)
        assert not set(train) & set(test)
        assert sorted(train + test) == sorted(f"c{i}" for i in range(12))
        assert split_test_cells([f"c{i}" for i in range(12)], 0.25) == (train, test)

    def test_test_cells_are_spread_out(self):
        """步长取样：测试电池在排序上均匀分布，不会恰好全落在同一批次。"""
        from run_e2e import split_test_cells

        cells = [f"c{i:02d}" for i in range(20)]
        _, test = split_test_cells(cells, 0.2)
        assert test == cells[::5], "应当每 5 颗取一颗，而不是取前 4 颗"
        assert test != cells[:4]

    def test_bad_fraction_is_refused(self):
        from run_e2e import split_test_cells

        for bad in (0.0, 1.0, -0.5):
            with pytest.raises(ValueError):
                split_test_cells([f"c{i}" for i in range(8)], bad)
