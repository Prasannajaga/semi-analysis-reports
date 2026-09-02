"""Shared runner process handling."""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Mapping, Sequence

from benchmark_tool.io import write_text


def run_process(
    command: Sequence[str],
    work_dir: Path,
    environment: Mapping[str, str] | None = None,
    log_prefix: str = "process",
    redact_values: Sequence[str] = (),
) -> dict[str, object]:
    started = time.time()
    child_environment = os.environ.copy()
    if environment:
        child_environment.update(environment)
    try:
        completed = subprocess.run(
            list(command),
            cwd=work_dir,
            env=child_environment,
            check=False,
            capture_output=True,
            text=True,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        return_code = completed.returncode
        error = None
    except OSError as exception:
        stdout = ""
        stderr = ""
        return_code = 127
        error = f"{type(exception).__name__}: {exception}"
    redactions = tuple(value for value in redact_values if value)
    stdout = _redact(stdout, redactions)
    stderr = _redact(stderr, redactions)
    error = _redact(error, redactions) if error is not None else None
    failure_summary = (
        _failure_summary(stderr, stdout, error) if return_code != 0 else None
    )
    write_text(work_dir / f"{log_prefix}-stdout.log", stdout)
    write_text(work_dir / f"{log_prefix}-stderr.log", stderr)
    return {
        "command": [_redact(argument, redactions) for argument in command],
        "returnCode": return_code,
        "durationSeconds": round(time.time() - started, 6),
        "error": error,
        "failureSummary": failure_summary,
    }


def _redact(value: str, secrets: Sequence[str]) -> str:
    redacted = value
    for secret in secrets:
        redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _failure_summary(stderr: str, stdout: str, error: str | None) -> str:
    if error:
        return error
    text = "\n".join(part for part in (stderr.strip(), stdout.strip()) if part)
    if not text:
        return "process exited without a diagnostic message"
    ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    lines = [ansi.sub("", line).strip() for line in text.splitlines() if line.strip()]
    preferred = (
        line
        for line in reversed(lines)
        if any(marker in line.lower() for marker in ("error", "failed", "exception", "429"))
        and not line.startswith("File ")
    )
    selected = next(preferred, lines[-1])
    return selected[:1000]
