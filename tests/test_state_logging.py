from __future__ import annotations

import sys
from copy import deepcopy

from benchmark_tool.config import BenchmarkConfig
from benchmark_tool.runners.common import run_process
from benchmark_tool.state_logging import configure_debug, log_error, log_state


def test_uppercase_debug_config_flag(config_dict):
    raw = deepcopy(config_dict)
    raw["DEBUG"] = True

    config = BenchmarkConfig.model_validate(raw)

    assert config.debug is True
    assert config.model_dump(mode="json", by_alias=True)["DEBUG"] is True


def test_debug_defaults_to_disabled(config_dict):
    config = BenchmarkConfig.model_validate(deepcopy(config_dict))
    assert config.debug is False


def test_state_logs_are_gated_but_errors_are_always_visible(capsys):
    configure_debug(False)
    log_state("config", "hidden detail")
    log_error("visible failure")
    assert capsys.readouterr().err == "[error] : visible failure\n"

    configure_debug(True)
    try:
        log_state("config", "visible detail")
        assert capsys.readouterr().err == "[config] : visible detail\n"
    finally:
        configure_debug(False)


def test_debug_process_output_is_streamed_and_redacted(capsys, tmp_path):
    secret = "terminal-secret-value"
    configure_debug(True)
    try:
        result = run_process(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    f"print('stdout {secret}'); "
                    f"print('stderr {secret}', file=sys.stderr)"
                ),
            ],
            tmp_path,
            log_prefix="test-runner",
            redact_values=(secret,),
        )
    finally:
        configure_debug(False)

    terminal = capsys.readouterr().err
    artifacts = (tmp_path / "test-runner-stdout.log").read_text() + (
        tmp_path / "test-runner-stderr.log"
    ).read_text()
    assert result["returnCode"] == 0
    assert "[test-runner:stdout] : stdout [REDACTED]" in terminal
    assert "[test-runner:stderr] : stderr [REDACTED]" in terminal
    assert secret not in terminal
    assert secret not in artifacts
