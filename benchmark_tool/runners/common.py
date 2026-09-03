"""Shared runner process handling."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO

from benchmark_tool.io import write_text
from benchmark_tool.state_logging import debug_enabled, log_state


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
    redactions = tuple(value for value in redact_values if value)
    safe_command = [_redact(argument, redactions) for argument in command]
    log_state("process", f"Starting: {shlex.join(safe_command)}")
    if environment:
        log_state("process", "Environment overrides: " + ", ".join(sorted(environment)))
    try:
        if debug_enabled():
            stdout, stderr, return_code = _run_streaming(
                command,
                work_dir,
                child_environment,
                log_prefix,
                redactions,
            )
        else:
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
    stdout = _redact(stdout, redactions)
    stderr = _redact(stderr, redactions)
    error = _redact(error, redactions) if error is not None else None
    failure_summary = (
        _failure_summary(stderr, stdout, error) if return_code != 0 else None
    )
    write_text(work_dir / f"{log_prefix}-stdout.log", stdout)
    write_text(work_dir / f"{log_prefix}-stderr.log", stderr)
    duration = round(time.time() - started, 6)
    completion = f"Finished with exit code {return_code} in {duration:.3f}s"
    if failure_summary:
        completion += f": {failure_summary}"
    log_state("process", completion)
    return {
        "command": safe_command,
        "returnCode": return_code,
        "durationSeconds": duration,
        "error": error,
        "failureSummary": failure_summary,
    }


def _run_streaming(
    command: Sequence[str],
    work_dir: Path,
    environment: Mapping[str, str],
    log_prefix: str,
    redactions: Sequence[str],
) -> tuple[str, str, int]:
    """Capture child output while also showing redacted lines in debug mode."""

    process = subprocess.Popen(
        list(command),
        cwd=work_dir,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        errors="replace",
    )
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    threads = [
        threading.Thread(
            target=_drain_stream,
            args=(process.stdout, stdout_parts, f"{log_prefix}:stdout", redactions),
            daemon=True,
        ),
        threading.Thread(
            target=_drain_stream,
            args=(process.stderr, stderr_parts, f"{log_prefix}:stderr", redactions),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    return_code = process.wait()
    for thread in threads:
        thread.join()
    return "".join(stdout_parts), "".join(stderr_parts), return_code


def _drain_stream(
    stream: IO[str] | None,
    destination: list[str],
    state: str,
    redactions: Sequence[str],
) -> None:
    if stream is None:
        return
    try:
        for line in stream:
            safe_line = _redact(line, redactions)
            destination.append(safe_line)
            log_state(state, safe_line.rstrip("\r\n"))
    finally:
        stream.close()


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
