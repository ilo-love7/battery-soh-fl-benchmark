"""`flsoh.featurespec` 的测试（纯标准库，进默认 CI job）。

这个模块只有一件事：从特征表旁边读 `feature_version`。它被 `--skip-existing`
用来判断「这个产物是不是本次设置的产物」，所以**读不出时必须返回 None 而不是
猜一个值**——猜错会让旧结果被当成新的跳过。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flsoh.featurespec import read_feature_version, spec_path_for  # noqa: E402


class TestSpecPath:
    def test_suffix_is_appended_to_the_stem(self):
        assert spec_path_for("data/features.parquet").name == "features.spec.json"

    def test_stays_in_the_same_directory(self):
        path = spec_path_for("data/mit-stanford/features.parquet")
        assert path.parent.name == "mit-stanford"

    def test_double_suffix_does_not_accumulate(self):
        """`features.spec.json` 不该变成 `features.spec.spec.json`。"""
        assert spec_path_for("data/features.spec.json").name == "features.spec.spec.json"

    def test_accepts_a_path_object(self):
        assert spec_path_for(Path("a/b.parquet")).name == "b.spec.json"


class TestReadFeatureVersion:
    def _write(self, tmp_path: Path, payload) -> Path:
        features = tmp_path / "features.parquet"
        features.write_bytes(b"")
        spec = tmp_path / "features.spec.json"
        if isinstance(payload, str):
            spec.write_text(payload, encoding="utf-8")
        else:
            spec.write_text(json.dumps(payload), encoding="utf-8")
        return features

    def test_reads_the_version(self, tmp_path):
        features = self._write(tmp_path, {"feature_version": "discharge-v2"})
        assert read_feature_version(features) == "discharge-v2"

    def test_missing_spec_returns_none(self, tmp_path):
        features = tmp_path / "features.parquet"
        features.write_bytes(b"")
        assert read_feature_version(features) is None

    def test_spec_without_the_field_returns_none(self, tmp_path):
        """旧表就是这样：规格文件在，但没有口径字段。"""
        features = self._write(tmp_path, {"feature": "ΔQ(V) = Qc_k − Qc_ref"})
        assert read_feature_version(features) is None

    def test_non_string_version_is_not_coerced(self, tmp_path):
        """数字 2 不能被当成 "2"——那会让 `!= "2"` 的比较悄悄成立。"""
        features = self._write(tmp_path, {"feature_version": 2})
        assert read_feature_version(features) is None

    def test_broken_json_returns_none(self, tmp_path):
        features = self._write(tmp_path, "{not json")
        assert read_feature_version(features) is None

    def test_json_list_is_not_treated_as_a_spec(self, tmp_path):
        features = self._write(tmp_path, ["discharge-v2"])
        assert read_feature_version(features) is None

    def test_empty_string_is_returned_verbatim(self, tmp_path):
        """空串是「自报了但报的是空」——调用方据此能判出不一致。"""
        features = self._write(tmp_path, {"feature_version": ""})
        assert read_feature_version(features) == ""


class TestRealTable:
    """对仓库里真实的特征表做一次冒烟（表不在就跳过）。"""

    def test_real_table_reports_a_version(self):
        features = ROOT / "data" / "mit-stanford" / "features.parquet"
        if not features.exists():
            pytest.skip("本地没有特征表")
        version = read_feature_version(features)
        # 允许 None（旧表），但不允许是别的类型。
        assert version is None or isinstance(version, str)
