"""Runtime BFCL registry bridge that preserves OpenRouter hard routing."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def load_routing() -> dict[str, Any]:
    value = json.loads(os.environ["BENCHMARK_OPENROUTER_ROUTING"])
    expected = {"only", "allow_fallbacks", "require_parameters"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("invalid BENCHMARK_OPENROUTER_ROUTING")
    return value


def main(config_path: Path) -> None:
    try:
        from bfcl_eval._llm_response_generation import main as generation_main
        from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
        from bfcl_eval.eval_checker.eval_runner import main as evaluation_main
        from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
    except ImportError as error:
        raise RuntimeError("BFCL is optional; install it with `uv sync --extra bfcl`") from error

    config = json.loads(config_path.read_text(encoding="utf-8"))
    routing = load_routing()

    class RoutedOpenRouterHandler(OpenAICompletionsHandler):  # type: ignore[misc, valid-type]
        def _build_client_kwargs(self) -> dict[str, Any]:
            kwargs = super()._build_client_kwargs()
            kwargs["timeout"] = config["timeoutSeconds"]
            return kwargs

        def generate_with_backoff(self, **kwargs: Any) -> Any:
            kwargs.pop("store", None)
            kwargs.setdefault("max_tokens", config["maxTokens"])
            kwargs.setdefault("seed", config["seed"])
            kwargs["extra_body"] = {"provider": routing}
            return super().generate_with_backoff(**kwargs)

    registry_name = config["registryName"]
    MODEL_CONFIG_MAPPING[registry_name] = ModelConfig(
        model_name=config["model"],
        display_name=registry_name,
        url="https://openrouter.ai/",
        org="OpenRouter",
        license="Provider dependent",
        model_handler=RoutedOpenRouterHandler,
        is_fc_model=True,
    )
    result_dir = Path(config["resultDir"])
    score_dir = Path(config["scoreDir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)
    if config.get("limit") is not None:
        raise ValueError("BFCL does not provide a stable row-limit option; use a smaller testCategory")
    generation_main(
        SimpleNamespace(
            model=[registry_name],
            test_category=[config["testCategory"]],
            temperature=config["temperature"],
            include_input_log=False,
            exclude_state_log=False,
            num_gpus=1,
            num_threads=1,
            gpu_memory_utilization=0.9,
            backend="sglang",
            skip_server_setup=True,
            local_model_path=None,
            result_dir=result_dir,
            allow_overwrite=False,
            run_ids=False,
            enable_lora=False,
            max_lora_rank=None,
            lora_modules=None,
        )
    )
    evaluation_main(
        [registry_name],
        [config["testCategory"]],
        result_dir,
        score_dir,
        partial_eval=False,
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m benchmark_tool.runners.bfcl_bridge CONFIG.json")
    main(Path(sys.argv[1]))
