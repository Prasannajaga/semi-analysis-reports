# Serverless Inference Endpoint Benchmark & Comparison

A comprehensive benchmark and analysis comparing public serverless LLM inference providers to evaluate user experience, latency, cost, correctness, and reliability, providing actionable recommendations based on customer priorities.

---

## Goal & Overview

The objective of this project is to benchmark public, serverless inference endpoints across leading providers and model architectures using an adapted **EndpointX Serverless Methodology**.

By evaluating real-world workloads, this analysis answers the fundamental question: **Why should a customer choose one provider over another based on their specific technical, economic, and operational priorities?**

## Development Setup

```bash
uv sync --locked
cp .env.example .env  # set OPENROUTER_API_KEY
```

> Optional correctness runners: `uv sync --locked --extra lm-eval --extra bfcl`

## Benchmark Workflow

```text
         benchmark.yaml
               │
               ▼
        ┌──────────────┐
        │ benchmark.py │
        └──────┬───────┘
               │
 ┌─────────────┴─────────────┐
 │                           │
 ▼                           ▼
Performance             Correctness
  AIPerf                Eval Runner
 │                           │
 ▼                           ▼
raw artifacts           eval artifacts
 │                           │
 └─────────────┬─────────────┘
               ▼
        ┌─────────────┐
        │ analysis.py │
        └──────┬──────┘
               │
 ┌─────────────┼────────────────┐
 │             │                │
 ▼             ▼                ▼
Performance  Reliability     Pricing
 │           Correctness        │
 │             │                │
 └─────────────┬────────────────┘
               ▼
         results.jsonl
               │
               ▼
            view.py
               │
               ▼
          report.html
```

## Quick Start

### 1. Dry Run & Preflight

```bash
# Offline check: validate YAML, generate configs & runner files without network
uv run python benchmark.py configs/provider-benchmark.example.yaml --dry-run

# Live check: probe endpoint compatibility & features with minimal 1-token requests
uv run python benchmark.py configs/provider-benchmark.example.yaml --preflight
```

### 2. Execute Benchmark

```bash
uv run python benchmark.py configs/provider-benchmark.example.yaml --output-dir results/
```

### 3. Normalize & View Report

```bash
# Normalize raw artifacts into canonical results.jsonl
uv run python analysis.py results/<run-id>

# Generate standalone HTML visualization report
uv run python view.py results/<run-id>/results.jsonl --output report.html
```

### 4. Tests

```bash
uv run pytest -m "not integration"
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
