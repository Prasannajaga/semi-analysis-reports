from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from benchmark_tool.combine import combine_results
from benchmark_tool.io import read_jsonl
from benchmark_tool.reporting.html import render_report
from benchmark_tool.results import CanonicalResult


FIXTURE = Path(__file__).parent / "fixtures" / "canonical" / "results.jsonl"


def test_canonical_jsonl_validation():
    raw = read_jsonl(FIXTURE)
    result = CanonicalResult.model_validate(raw[0])
    assert result.phase == "performance"
    assert result.agentx is not None and result.agentx.submission_valid is True


def test_combine_deduplicates_exact_records(tmp_path):
    output = tmp_path / "combined.jsonl"
    records = combine_results([FIXTURE, FIXTURE], output)
    assert len(records) == 1
    assert len(read_jsonl(output)) == 1


def test_combine_rejects_conflicting_duplicate(tmp_path):
    raw = read_jsonl(FIXTURE)[0]
    raw["performance"]["ttft"]["p95"] = 999
    conflict = tmp_path / "conflict.jsonl"
    conflict.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting duplicate"):
        combine_results([FIXTURE, conflict], tmp_path / "combined.jsonl")


def test_combine_sorts_without_aggregating(tmp_path):
    first = read_jsonl(FIXTURE)[0]
    second = json.loads(json.dumps(first))
    second["run_id"] = "run-0"
    second["job_id"] = "job-0"
    unordered = tmp_path / "unordered.jsonl"
    unordered.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
    records = combine_results([unordered], tmp_path / "combined.jsonl")
    assert [record.run_id for record in records] == ["run-0", "run-1"]
    assert records[1].performance == CanonicalResult.model_validate(first).performance


def test_view_generation_is_self_contained(tmp_path):
    input_path = tmp_path / "results.jsonl"
    shutil.copy(FIXTURE, input_path)
    output = tmp_path / "report.html"
    render_report(input_path, output)
    html = output.read_text(encoding="utf-8")
    assert "OpenRouter Provider Benchmark" in html
    assert "model-a" in html
    assert "https://cdn" not in html
    assert "<script>" in html
