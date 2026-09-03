"""Small terminal logger for benchmark execution state."""

from __future__ import annotations

import sys
import threading

_debug_enabled = False
_output_lock = threading.Lock()


def configure_debug(enabled: bool) -> None:
    """Enable or disable detailed execution logging for the current process."""

    global _debug_enabled
    _debug_enabled = enabled


def debug_enabled() -> bool:
    return _debug_enabled


def log_state(state: str, message: object, *, always: bool = False) -> None:
    """Write one or more consistently formatted state lines to stderr."""

    if not (always or _debug_enabled):
        return
    lines = str(message).splitlines() or [""]
    with _output_lock:
        for line in lines:
            print(f"[{state}] : {line}", file=sys.stderr, flush=True)


def log_error(message: object) -> None:
    """Errors are always visible, even when detailed logging is disabled."""

    log_state("error", message, always=True)
