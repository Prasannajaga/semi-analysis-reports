# Serverless Inference Endpoint Benchmark & Comparison

A comprehensive benchmark and analysis comparing public serverless LLM inference providers to evaluate user experience, latency, cost, correctness, and reliability, providing actionable recommendations based on customer priorities.

---

## Goal & Overview

The objective of this project is to benchmark public, serverless inference endpoints across leading providers and model architectures using an adapted **EndpointX Serverless Methodology**.

By evaluating real-world workloads, this analysis answers the fundamental question: **Why should a customer choose one provider over another based on their specific technical, economic, and operational priorities?**

## Development setup

This project uses [uv](https://docs.astral.sh/uv/) with Python 3.12. The lockfile is
the source of truth for reproducible installs.

```bash
uv sync --locked
cp .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env` before running live OpenRouter requests. The
default environment includes AIPerf and the development tools. Correctness runners
are optional because they add large dependency trees:

```bash
# Install one runner:
uv sync --locked --extra lm-eval
uv sync --locked --extra bfcl

# Or install both runners:
uv sync --locked --all-extras
```

The current BFCL distribution pulls PyTorch and CUDA runtime packages, so its
optional environment requires several gigabytes of download and disk space.

Run project commands inside the managed environment:

```bash
uv run aiperf --version
uv run pytest
uv run ruff check .
uv run mypy
```

## OpenRouter benchmark workflow

The implementation deliberately keeps execution, normalization, and presentation
separate:

```text
configs/provider-benchmark.example.yaml
  -> benchmark.py  -> immutable raw run directory
  -> analysis.py   -> canonical results.jsonl
  -> view.py       -> self-contained report.html
```

- Performance is measured by AIPerf 0.12. The full example uses its
  `inferencex-agentx-mvp` scenario; the focused routing check uses a short
  synthetic workload.
- Reliability and SLO checks are derived from those same AIPerf profiling
  request records; there is no reliability workload.
- Pricing is derived from measured token usage and the OpenRouter endpoint price
  snapshot captured before execution; there is no pricing workload.
- Correctness is executed independently by lm-eval or BFCL.

Every request body contains one top-level OpenRouter routing object with exactly
one provider, `allow_fallbacks: false`, and `require_parameters: true`. A live run
preflights each model/provider pair first. An endpoint that cannot preserve the
selected workload semantics is recorded as `unsupported` and is not benchmarked.

### Configure

Copy the example if you want to change models, providers, tasks, or concurrency:

```bash
cp configs/provider-benchmark.example.yaml configs/provider-benchmark.yaml
```

Tokenizer IDs are explicit because AIPerf needs accurate token accounting. The
example uses the current date-pinned AIPerf dataset
`semianalysis_cc_traces_weka_062126`; AIPerf remains the final authority and
validates every generated native configuration. Export the key or set it in the
local `.env` file (which is ignored by Git):

```bash
export OPENROUTER_API_KEY=...
```

For a Hugging Face tokenizer that requires repository-defined Python code,
review that repository first and opt in explicitly on that model:

```yaml
tokenizer: moonshotai/Kimi-K3
tokenizerTrustRemoteCode: true
```

This becomes AIPerf's native `tokenizer.trustRemoteCode` setting. It is `false`
by default.

The secret is passed only through the child-process environment. Generated
runner configs, manifests, logs, canonical JSONL, and HTML never contain it.

### Dry run

A dry run validates the user YAML, expands all jobs, writes the complete raw
directory structure and runner configs, and runs AIPerf's native config validator.
It makes no OpenRouter requests:

```bash
uv run python benchmark.py configs/provider-benchmark.example.yaml --dry-run
```

To make only the minimal live compatibility checks, explicitly use:

```bash
uv run python benchmark.py configs/provider-benchmark.example.yaml --preflight
```

For a focused one-model/one-provider routing check, use the smaller test
configuration. It pins DeepInfra, disables fallback, requires all request
parameters, checks streaming with a simple synthetic performance workload,
and does not use AgentX or execute the benchmark workload:

```bash
uv run python benchmark.py configs/openrouter-routing-test.yaml --preflight
```

Inspect the printed run directory's
`models/qwen3-32b/providers/deepinfra/endpoint/preflight.json`. A successful
check records `status: supported`; `routingVerified: true` means OpenRouter
also exposed provider metadata that matched `deepinfra`.

### Execute

Install the correctness extras selected by the YAML, then execute the matrix:

```bash
uv sync --locked --extra lm-eval --extra bfcl
uv run python benchmark.py configs/provider-benchmark.example.yaml --output-dir results/
```

AgentX runs for at least 900 seconds and may download its public trace corpus and
tokenizer on first use. BFCL is intentionally optional because its dependency
set is several gigabytes.

### Analyze, combine, and view

Normalization reads structured runner artifacts, never terminal tables:

```bash
uv run python analysis.py results/<run-id>
```

This creates `results/<run-id>/results.jsonl`, including explicit failed,
unsupported, and planned jobs. Combining validates and orders records without
averaging, ranking, or changing any metric:

```bash
uv run python analysis.py \
  --combine \
  results/run-a/results.jsonl \
  results/run-b/results.jsonl \
  --output combined-results.jsonl
```

Render a standalone report from canonical JSONL only:

```bash
uv run python view.py combined-results.jsonl --output report.html
```

### Tests

The suite uses fixture artifacts and mocked HTTP responses. It does not require
paid API calls:

```bash
uv run pytest -m "not integration"
```

The one-token live check requires both the key and explicit authorization:

```bash
RUN_OPENROUTER_INTEGRATION=1 uv run pytest -m integration tests/test_live_openrouter.py
```

## Evaluation Criteria

The benchmark evaluates endpoints across five core dimensions:

### 1. Latency & Interactivity
- **Time to First Token (TTFT):** Pre-fill latency and streaming responsiveness across varying prompt lengths.
- **Inter-Token Latency (ITL):** Time between consecutive generated tokens (P50, P95, P99 jitter and generation cadence).
- **End-to-End Latency (E2EL):** Total elapsed time from request initiation to completion.
- **Interactivity:** Streaming smoothness and user-perceived responsiveness.
- **Network Latency Isolation:** Measuring and stripping out client-to-server network latency and round-trip time (RTT) to accurately isolate server-side endpoint performance.

### 2. Cost & Efficiency
- **Cost per Token:** Baseline pricing for input, output, and cached tokens ($/M tokens).
- **Cost per Task:** Total dollar cost required to execute benchmark tasks.
- **Correctness-Adjusted Cost:** Effective cost per successful task completion, factoring in wasted spend from incorrect, malformed, or retried generations.
- **Cache-Adjusted Cost:** Effective cost taking into account observed cache hit ratios across multi-turn conversations and repeated prompt prefixes.

### 3. Correctness & Output Quality
- **Task Accuracy & Fidelity:** Evaluation of answer correctness against ground truth across standardized reasoning, coding, and instruction-following benchmarks.
- **Quantization & Optimization Impact:** Detecting output quality degradation, reasoning regression, or formatting drift caused by provider-side quantization or kernel optimizations (e.g., FP8 vs. FP16).
- **Structured Output Compliance:** Schema validation and reliability when generating strict JSON outputs and executing function/tool calls.

### 4. Reliability & Response Integrity
- **Bad Response Frequency:** Tracking the occurrence of degenerate outputs, including empty responses, premature stop cutoffs, repetitive generation loops, and malformed content.
- **Error & Failure Rates:** Frequency of HTTP 5xx server errors, request timeouts, connection drops, and 429 rate limit throttling under load.
- **Response Consistency:** Stability and variance of completions across repeated runs under identical inputs.

### 5. Cache Hit Rate & Prefix Caching
- **Cache Hit Efficiency:** KV cache and prompt prefix cache detection rates for repeated contexts (system prompts, multi-turn chat history, RAG documents).
- **TTFT Reduction via Caching:** Pre-fill acceleration and latency reduction observed on cache hits versus cache misses.
- **Realized Cost Savings:** Direct financial impact of cache discounts based on observed hit ratios.

---

## Deliverables & Artifacts

- **Benchmarking Suite:** Reproducible test harness measuring provider-pinned serverless metrics through OpenRouter.
- **Empirical Dataset:** Comprehensive trace logs covering TTFT, ITL, E2EL, per-task costs, correctness scores, and error/bad response logs.
- **Visualizations & Dashboard:** Interactive charts and reports visualizing latency distributions, correctness-adjusted costs, and provider reliability comparisons.
- **Customer Decision Matrix:** Prescriptive recommendations mapping customer use cases (e.g., interactive real-time applications, cost-sensitive batch processing, high-fidelity agentic workflows) to the ideal provider.
