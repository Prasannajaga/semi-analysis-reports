"""Validate, de-duplicate, and deterministically combine canonical JSONL."""

from __future__ import annotations

from pathlib import Path

from benchmark_tool.io import canonical_json, read_jsonl, write_jsonl
from benchmark_tool.results import CanonicalResult


def sort_key(record: CanonicalResult) -> tuple[object, ...]:
    return (
        record.run_id,
        record.model.id,
        record.provider.id,
        record.phase,
        record.workload.name,
        record.workload.concurrency or 0,
        record.job_id,
    )


def combine_results(inputs: list[Path], output: Path) -> list[CanonicalResult]:
    identities: dict[tuple[str, str], CanonicalResult] = {}
    for path in inputs:
        for raw in read_jsonl(path):
            record = CanonicalResult.model_validate(raw)
            identity = (record.run_id, record.job_id)
            previous = identities.get(identity)
            if previous is not None and canonical_json(previous.json_record()) != canonical_json(record.json_record()):
                raise ValueError(f"conflicting duplicate canonical result: {identity}")
            identities[identity] = record
    records = sorted(identities.values(), key=sort_key)
    write_jsonl(output, (record.json_record() for record in records))
    return records
