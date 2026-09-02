#!/usr/bin/env python3
"""Normalize raw benchmark artifacts or combine canonical JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_tool.combine import combine_results
from benchmark_tool.normalization import normalize_run


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("run", nargs="?", type=Path, help="raw run directory to normalize")
    value.add_argument(
        "--combine",
        nargs="+",
        type=Path,
        metavar="RESULTS_JSONL",
        help="validate, deduplicate, sort, and combine canonical JSONL files",
    )
    value.add_argument("--output", type=Path)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.combine:
        output = args.output or Path("combined-results.jsonl")
        combine_results(args.combine, output)
    else:
        if args.run is None:
            parser().error("provide a run directory or --combine JSONL files")
        output = args.output or args.run / "results.jsonl"
        normalize_run(args.run, output)
    print(output.resolve())


if __name__ == "__main__":
    main()
