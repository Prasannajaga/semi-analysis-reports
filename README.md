# LLM Provider Benchmark Report

Reports and benchmarks to determine which inference providers deliver the best performance, throughput, and cost efficiency.

## Goal

The goal of this benchmark is to evaluate and compare a single model across three different inference providers to determine which provider performs best in terms of:

- **Throughput:** Output tokens generated per second (tokens/s) under varying concurrency levels.
- **Latency:** Time to First Token (TTFT) and overall request turnaround time.
- **Cost Efficiency:** Price-to-performance ratio (throughput per dollar).
- **Correctness & Output Quality:** Answer accuracy, formatting fidelity, and consistency across providers (ensuring optimizations or quantization don't degrade output quality).
- **Cache Hit Rate & Prefix Caching:** Prompt/KV cache hit behavior, latency reductions on cached prefixes, and effective cost savings under repeated or multi-turn workloads.

By running standardized benchmarks across all three providers under identical prompt sizes and workloads, this study identifies the most performant and cost-effective hosting solution.
