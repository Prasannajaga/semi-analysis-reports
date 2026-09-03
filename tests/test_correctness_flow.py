from __future__ import annotations

import sys
from pathlib import Path

from benchmark_tool.config import load_config
from benchmark_tool.matrix import expand_matrix
from benchmark_tool.runners import lm_eval

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "correctness-smoke-test.yaml"


def test_smoke_config_creates_only_one_limited_correctness_job():
    config = load_config(SMOKE_CONFIG)
    jobs = expand_matrix(config)

    assert config.phases.performance is None
    assert config.phases.correctness is not None
    assert config.phases.correctness.enabled is True
    assert config.phases.pricing.enabled is False
    assert config.phases.reliability.enabled is False
    assert len(jobs) == 1
    assert jobs[0].phase == "correctness"
    assert jobs[0].runner == "lm-eval"
    assert jobs[0].task_name == "gsm8k"
    assert jobs[0].limit == 5


def test_smoke_config_generates_direct_lm_eval_configuration(tmp_path):
    config = load_config(SMOKE_CONFIG)
    job = expand_matrix(config)[0]

    native = lm_eval.make_lm_eval_config(config, job, tmp_path / "artifacts")

    assert native["model"] == "local-chat-completions"
    assert native["model_args"]["model"] == "moonshotai/Kimi-K3"
    assert native["model_args"]["base_url"] == (
        "https://api.together.ai/v1/chat/completions"
    )
    assert native["model_args"]["eos_string"] == "<|im_end|>"
    assert native["tasks"] == ["gsm8k"]
    assert native["limit"] == 5
    assert native["gen_kwargs"] == {"temperature": 0.0, "max_tokens": 512}


def test_lm_eval_execution_uses_active_python_and_passes_key_only_in_environment(
    monkeypatch, tmp_path
):
    config = load_config(SMOKE_CONFIG)
    job = expand_matrix(config)[0]
    captured = {}

    def fake_run_process(
        command,
        work_dir,
        environment=None,
        log_prefix="process",
        redact_values=(),
    ):
        captured.update(
            command=command,
            work_dir=work_dir,
            environment=environment,
            log_prefix=log_prefix,
            redact_values=redact_values,
        )
        return {"returnCode": 0, "failureSummary": None}

    monkeypatch.setattr(lm_eval, "run_process", fake_run_process)

    result = lm_eval.execute(config, job, tmp_path, "test-secret")

    assert result["status"] == "completed"
    assert result["reason"] is None
    assert captured["command"] == [
        sys.executable,
        "-m",
        "lm_eval",
        "run",
        "--config",
        str(tmp_path / "lm-eval-config.yaml"),
    ]
    assert captured["environment"] == {"OPENAI_API_KEY": "test-secret"}
    assert captured["redact_values"] == ("test-secret",)
    assert "test-secret" not in (tmp_path / "lm-eval-config.yaml").read_text()


def test_lm_eval_execution_reports_runner_failure(monkeypatch, tmp_path):
    config = load_config(SMOKE_CONFIG)
    job = expand_matrix(config)[0]

    monkeypatch.setattr(
        lm_eval,
        "run_process",
        lambda *args, **kwargs: {
            "returnCode": 2,
            "failureSummary": "invalid evaluation configuration",
        },
    )

    result = lm_eval.execute(config, job, tmp_path, "test-secret")

    assert result["status"] == "failed"
    assert result["reason"] == (
        "lm-eval exited with code 2: invalid evaluation configuration"
    )
