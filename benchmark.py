#!/usr/bin/env python3
"""Execute a reproducible OpenRouter provider benchmark matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_tool.execution import execute_benchmark
from benchmark_tool.io import read_json


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("config", type=Path, help="strict benchmark YAML")
    value.add_argument(
        "--output-dir",
        "--output",
        dest="output",
        type=Path,
        default=Path("results"),
        help="raw run root",
    )
    value.add_argument(
        "--dry-run",
        action="store_true",
        help="generate the full matrix and runner configs without inference",
    )
    value.add_argument(
        "--preflight",
        action="store_true",
        help="generate configs and make only the minimal live endpoint preflight request",
    )
    return value


def main() -> None:
    args = parser().parse_args()
    path = execute_benchmark(
        args.config,
        args.output,
        dry_run=args.dry_run or args.preflight,
        preflight_during_dry_run=args.preflight,
    )
    manifest = read_json(path / "manifest.json")
    print(path)
    for job in manifest["jobs"]:
        reason = f" — {job['reason']}" if job.get("reason") else ""
        print(f"{job['status']:>11}  {job['jobId']}{reason}")
        if job["status"] in {"failed", "unsupported"}:
            print(f"             details: {path / job['path'] / 'job.json'}")


if __name__ == "__main__":
    main()
