#!/usr/bin/env python3
"""Normalize raw benchmark artifacts or combine canonical JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from benchmark_tool.combine import combine_results
from benchmark_tool.normalization import normalize_run


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "run",
        nargs="*",
        type=Path,
        help="raw run directory (or directories) to normalize",
    )
    value.add_argument(
        "--combine",
        nargs="+",
        type=Path,
        metavar="RESULTS_JSONL",
        help="validate, deduplicate, sort, and combine canonical JSONL files",
    )
    value.add_argument(
        "--output",
        type=Path,
        help="output path for normalized or combined JSONL",
    )
    return value


def main() -> None:
    args = parser().parse_args()
    if args.combine:
        output = args.output or Path("combined-results.jsonl")
        combine_results(args.combine, output)
        print(output.resolve())
        return

    runs = args.run if isinstance(args.run, list) else ([args.run] if args.run else [])
    if not runs:
        parser().error("provide at least one run directory or --combine JSONL files")

    if len(runs) == 1:
        run_path = runs[0]
        if run_path.is_file() and run_path.name.endswith(".jsonl"):
            output = args.output or run_path
            if output != run_path:
                shutil.copy(run_path, output)
            print(output.resolve())
            return
        output = args.output or run_path / "results.jsonl"
        normalize_run(run_path, output)
        print(output.resolve())
    else:
        generated: list[Path] = []
        for run_path in runs:
            if run_path.is_file() and run_path.name.endswith(".jsonl"):
                generated.append(run_path)
            else:
                jsonl_path = run_path / "results.jsonl"
                normalize_run(run_path, jsonl_path)
                generated.append(jsonl_path)
        output = args.output or Path("combined-results.jsonl")
        combine_results(generated, output)
        print(output.resolve())


if __name__ == "__main__":
    main()
