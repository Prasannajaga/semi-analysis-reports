#!/usr/bin/env python3
"""Execute a reproducible OpenRouter provider benchmark matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import ValidationError

from benchmark_tool.execution import execute_benchmark
from benchmark_tool.io import read_json
from benchmark_tool.state_logging import log_error


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


def _clean_error(error: Exception) -> str:
    if isinstance(error, ValidationError):
        details = []
        for issue in error.errors(include_url=False):
            location = ".".join(str(part) for part in issue["loc"])
            details.append(f"{location}: {issue['msg']}")
        return "Configuration validation failed:\n" + "\n".join(
            f"- {detail}" for detail in details
        )
    if isinstance(error, FileNotFoundError):
        return f"File not found: {error.filename or error}"
    if isinstance(error, PermissionError):
        return f"Permission denied: {error.filename or error}"
    message = str(error).strip() or "no diagnostic message was provided"
    return f"{type(error).__name__}: {message}"


def main() -> int:
    args = parser().parse_args()
    try:
        path = execute_benchmark(
            args.config,
            args.output,
            dry_run=args.dry_run or args.preflight,
            preflight_during_dry_run=args.preflight,
        )
    except KeyboardInterrupt:
        log_error("Benchmark interrupted by user")
        return 130
    except Exception as error:
        log_error(_clean_error(error))
        return 1
    manifest = read_json(path / "manifest.json")
    print(f"[output] : {path}")
    for job in manifest["jobs"]:
        reason = f" — {job['reason']}" if job.get("reason") else ""
        print(f"[{job['status']}] : {job['jobId']}{reason}")
        if job["status"] in {"failed", "unsupported"}:
            print(f"[details] : {path / job['path'] / 'job.json'}")
    print(f"[summary] : {manifest['status']} ({len(manifest['jobs'])} jobs)")
    if (path / "results.jsonl").exists():
        print(f"[results] : {path / 'results.jsonl'}")
    return 1 if manifest["status"] in {"failed", "completed-with-errors"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
