#!/usr/bin/env python3
"""Render a self-contained HTML report from canonical results JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_tool.reporting.html import render_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="canonical results.jsonl")
    parser.add_argument("--output", type=Path, default=Path("report.html"))
    args = parser.parse_args()
    render_report(args.results, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
