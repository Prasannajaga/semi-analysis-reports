# Serverless Inference Endpoint Benchmark & Comparison

A comprehensive benchmark and analysis comparing public serverless LLM inference providers to evaluate user experience, latency, cost, correctness, and reliability, providing actionable recommendations based on customer priorities.

---

## Goal & Overview

The objective of this project is to benchmark public, serverless inference endpoints across leading providers and model architectures using an adapted **EndpointX Serverless Methodology**.

By evaluating real-world workloads, this analysis answers the fundamental question: **Why should a customer choose one provider over another based on their specific technical, economic, and operational priorities?**
 
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

Set the top-level `DEBUG` and `normalize` flags in the benchmark YAML:

```yaml
DEBUG: true      # detailed [state] : message execution logs and live runner output
normalize: true  # automatically generate canonical results.jsonl upon run completion
```

Use `DEBUG: false` (the default when omitted) to show only the final job summary and
errors. API keys are redacted from both terminal output and runner log artifacts.

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
If `normalize: true` is configured in the YAML, canonical `results.jsonl` is generated automatically in `results/<run-id>/results.jsonl`.

### 3. Normalize & View Report

```bash
# Normalize raw artifacts into canonical results.jsonl (if not using normalize: true)
uv run python analysis.py results/<run-id>

# Generate standalone HTML visualization report
uv run python view.py results/<run-id>/results.jsonl --output report.html
```

### 4. Combine Multiple Runs

Validate, deduplicate, sort, and combine multiple benchmark runs into a unified canonical dataset:

```bash
# Combine existing canonical results.jsonl files
uv run python analysis.py --combine results/run-1/results.jsonl results/run-2/results.jsonl --output combined-results.jsonl

# Or normalize and combine multiple raw run directories directly
uv run python analysis.py results/run-1 results/run-2 --output combined-results.jsonl

# Generate HTML report from combined results
uv run python view.py combined-results.jsonl --output combined-report.html
```

### 5. Tests

```bash
uv run pytest
```
 