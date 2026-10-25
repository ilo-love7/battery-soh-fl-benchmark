"""Schema v1 校验的测试。

为什么值得单独测：这个校验器是 App 与联邦仿真之间**唯一**的契约闸门。
它漏放一条坏会话，坏数据会一路流到特征表和指标里，而指标看不出任何异常
（时间戳重复只会让 Q(V) 整体偏高，曲线形状不变）。所以每条边界都要有一个
「放过了会怎样」的负控，而不是只测「合法的能过」。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from flsoh import schema as sc

REPO_ROOT = Path(__file__).resolve().parents[1]


def _session(**over):
    s = {
        "user_id_hash": "a" * 32,
        "battery_id": "bat-0001",
        "session_id": "ses-0001",
        "start_ts": 1_700_000_000,
        "end_ts": 1_700_000_000 + 3600,
        "chemistry_hint": "unknown",
        "sampling_hint": "variable",
        "records": [
            {"t": 1_700_000_000 + i * 10, "V": 3.7, "I": -1.2,
             "T": 29.5, "soc_est": 0.4}
            for i in range(20)
        ],
    }
    s.update(over)
    return s


class TestValid:
    def test_minimal_session_passes(self):
        assert sc.validate_session(_session()) == []

    def test_explicit_version_passes(self):
        assert sc.validate_session(_session(schema_version="v1")) == []

    def test_unknown_fields_only_warn(self):
        """v2 会加字段；现在报错等于宣告这个仓库只认 v1。"""
        errs = sc.validate_session(_session(future_field=1))
        assert errs == []
        assert sc.validate_session(_session(future_field=1), strict=True)

    def test_all_chemistry_hints_pass(self):
        for hint in sc.CHEMISTRY_HINTS:
            assert sc.validate_session(_session(chemistry_hint=hint)) == []


class TestVersionMismatch:
    def test_wrong_version_is_an_error(self):
        """结构合法、语义相反是最危险的一类错误。"""
        errs = sc.validate_session(_session(schema_version="v2"))
        assert errs and "schema_version" in errs[0]


class TestRequiredFields:
    @pytest.mark.parametrize("field", sc.SESSION_FIELDS)
    def test_missing_field_is_reported(self, field):
        s = _session()
        del s[field]
        errs = sc.validate_session(s)
        assert errs and field in errs[0]

    def test_missing_field_stops_early(self):
        """缺字段时不要连锁报一堆错——第一条之后没有可检查的东西。"""
        s = _session()
        del s["records"]
        del s["start_ts"]
        assert len(sc.validate_session(s)) == 1


class TestIdentity:
    def test_hash_must_be_hex(self):
        errs = sc.validate_session(_session(user_id_hash="not-a-hash"))
        assert errs and "user_id_hash" in errs[0]

    def test_hash_too_short(self):
        errs = sc.validate_session(_session(user_id_hash="abc"))
        assert errs and "user_id_hash" in errs[0]

    def test_empty_id_is_rejected(self):
        assert sc.validate_session(_session(session_id=""))

    def test_oversized_id_is_rejected(self):
        assert sc.validate_session(_session(battery_id="x" * 200))


class TestTimestamps:
    def test_inverted_window(self):
        errs = sc.validate_session(_session(end_ts=1))
        assert errs and "end_ts" in errs[0]

    def test_seconds_milliseconds_confusion(self):
        """把秒写成毫秒：窗口变成 41 天，超过 7 天上限。"""
        errs = sc.validate_session(
            _session(start_ts=1_700_000_000_000, end_ts=1_700_000_003_600_000)
        )
        assert errs and "不在" in errs[0]

    def test_records_outside_window(self):
        """会话被拼进别的时间窗，按时间切 train/val 会切错。"""
        s = _session()
        s["records"][-1]["t"] = s["end_ts"] + 999
        errs = sc.validate_session(s)
        assert errs and "会话窗口" in errs[-1]

    def test_broken_timestamp_does_not_chain(self):
        """start_ts 不是数时，不要再去比对 records 的时间窗。"""
        s = _session(start_ts="oops")
        errs = sc.validate_session(s)
        assert len(errs) == 1


class TestRecords:
    def test_empty_records(self):
        assert sc.validate_session(_session(records=[]))

    def test_non_increasing_time(self):
        """时间不递增 → 容量积分累加重复区间 → Q(V) 整体偏高而形状不变。"""
        s = _session()
        s["records"][5]["t"] = s["records"][4]["t"]
        errs = sc.validate_session(s)
        assert errs and "未严格大于" in errs[0]

    def test_voltage_out_of_physical_range(self):
        s = _session()
        s["records"][3]["V"] = 12.0
        errs = sc.validate_session(s)
        assert errs and ".V=" in errs[0]

    def test_current_out_of_range(self):
        s = _session()
        s["records"][3]["I"] = -50.0
        assert sc.validate_session(s)

    def test_temperature_out_of_range(self):
        s = _session()
        s["records"][3]["T"] = 200.0
        assert sc.validate_session(s)

    def test_soc_not_divided_by_100(self):
        """手机把 42% 写成 42： soc_est 越界，一眼能看出来。"""
        s = _session()
        s["records"][3]["soc_est"] = 42
        errs = sc.validate_session(s)
        assert errs and "soc_est" in errs[0]

    def test_nan_is_rejected(self):
        s = _session()
        s["records"][3]["V"] = float("nan")
        assert sc.validate_session(s)

    def test_bool_is_not_a_number(self):
        """``True`` 是 ``int`` 的子类，但把它当 1.0 用一定是字段错位。"""
        s = _session()
        s["records"][3]["V"] = True
        assert sc.validate_session(s)

    def test_record_missing_field(self):
        s = _session()
        del s["records"][7]["T"]
        errs = sc.validate_session(s)
        assert errs and "records[7]" in errs[0]

    def test_too_many_records(self):
        s = _session(records=[{"t": i, "V": 3.7, "I": -1.0, "T": 25.0,
                               "soc_est": 0.5} for i in range(sc.MAX_RECORDS + 1)])
        assert sc.validate_session(s)


class TestBatch:
    def test_counts(self):
        good = [_session() for _ in range(3)]
        bad = [_session(session_id="")]
        rep = sc.validate_sessions(good + bad)
        assert rep == {**rep, "total": 4, "valid": 3, "invalid": 1}
        assert rep["sample_errors"][0]["index"] == 3

    def test_sample_errors_is_capped(self):
        """不要把几千条错误全塞进返回值。"""
        rep = sc.validate_sessions([_session(session_id="") for _ in range(50)])
        assert len(rep["sample_errors"]) == 20


class TestFingerprint:
    def test_same_content_same_fingerprint(self):
        assert sc.session_fingerprint(_session()) == sc.session_fingerprint(_session())

    def test_key_order_does_not_matter(self):
        """指纹必须跨进程稳定，否则去重没有意义。"""
        a = _session()
        b = dict(reversed(list(a.items())))
        assert sc.session_fingerprint(a) == sc.session_fingerprint(b)

    def test_one_changed_record_changes_it(self):
        a = _session()
        b = _session()
        b["records"][3]["V"] = 3.71
        assert sc.session_fingerprint(a) != sc.session_fingerprint(b)

    def test_only_id_same_but_content_differs(self):
        """重装 App 后 session_id 可能重复，内容不同不能算重复。"""
        a = _session()
        b = _session()
        b["records"] = b["records"][:5]
        assert sc.session_fingerprint(a) != sc.session_fingerprint(b)

    def test_canonical_json_handles_types(self):
        assert sc.canonical_json({"b": 1, "a": 2.5}) == '{"a":2.5,"b":1}'
        assert sc.canonical_json([1, True, None, "x"]) == '[1,true,null,"x"]'


class TestCheckerCli:
    """check_schema.py 的行为：退出码是入库脚本唯一能看的东西。

    检查器必须有负控（`tests/test_summarize.py` 同款理由）：永远 PASS 的
    检查器比没有检查器更糟。
    """

    @staticmethod
    def _write(tmp_path, sessions, *, raw=None):
        p = tmp_path / "sessions.jsonl"
        if raw is not None:
            p.write_text(raw, encoding="utf-8")
        else:
            p.write_text(
                "\n".join(json.dumps(s) for s in sessions) + "\n",
                encoding="utf-8",
            )
        return p

    def test_all_valid_exits_zero(self, tmp_path):
        root = REPO_ROOT
        p = self._write(tmp_path, [_session() for _ in range(3)])
        r = subprocess.run(
            [sys.executable, str(root / "scripts" / "check_schema.py"), str(p)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stdout + r.stderr
        assert "PASS" in r.stdout

    def test_one_invalid_exits_one(self, tmp_path):
        root = REPO_ROOT
        p = self._write(tmp_path, [_session(), _session(session_id="")])
        r = subprocess.run(
            [sys.executable, str(root / "scripts" / "check_schema.py"), str(p)],
            capture_output=True, text=True,
        )
        assert r.returncode == 1
        assert "FAIL" in r.stdout

    def test_malformed_json_exits_two(self, tmp_path):
        root = REPO_ROOT
        p = self._write(tmp_path, [], raw="{not json}\n")
        r = subprocess.run(
            [sys.executable, str(root / "scripts" / "check_schema.py"), str(p)],
            capture_output=True, text=True,
        )
        assert r.returncode == 2
        assert "不是合法 JSON" in r.stdout

    def test_missing_file_exits_two(self, tmp_path):
        root = REPO_ROOT
        r = subprocess.run(
            [sys.executable, str(root / "scripts" / "check_schema.py"),
             str(tmp_path / "nope.jsonl")],
            capture_output=True, text=True,
        )
        assert r.returncode == 2
        assert "不存在" in r.stdout
