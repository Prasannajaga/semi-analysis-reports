"""Self-contained HTML rendering from canonical JSONL only."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark_tool.io import read_jsonl, write_text
from benchmark_tool.results import CanonicalResult

TEMPLATE_PATH = Path(__file__).parent / "template.html"


def render_report(input_path: Path, output_path: Path) -> list[CanonicalResult]:
    records = [CanonicalResult.model_validate(raw) for raw in read_jsonl(input_path)]
    data = json.dumps([record.json_record() for record in records], ensure_ascii=False).replace("<", "\\u003c")
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    write_text(output_path, template.replace("__DATA__", data))
    return records
