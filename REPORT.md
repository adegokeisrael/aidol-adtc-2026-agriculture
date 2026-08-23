# Technical Report — AIDOL Agricultural Advisory Assistant

**Team ID:** AIDOL
**Submitter:** Adegoke Israel Adedolapo
**Domain:** agriculture
**Model:** AIDOL-Agri-Advisor-1.5B-Q4_K_M

---

## Problem

Millions of smallholder farmers and agricultural extension officers across Nigeria operate with unreliable internet, expensive mobile data, and no practical access to cloud-hosted AI advisory tools. This model is an offline, single-turn advisory assistant covering crop diagnosis, fertilizer/input calculation, pest management, and market economics, designed to run entirely on an 8GB commodity laptop with no cloud dependency.

The system is stateless by design — each prompt is self-contained (all needed context, such as quantities, symptoms, or region, is embedded in the question itself), matching both realistic field usage and the structure of the competition's own accuracy evaluation.

---

## Design Decisions

- **Base model:** Qwen2.5-1.5B-Instruct — Apache 2.0 licensed, strong instruction-following relative to size, mature GGUF/llama.cpp conversion support, and a resource footprint that leaves substantial headroom under the 7GB memory budget while retaining enough capacity for genuine domain competence.
- **Training pipeline:** knowledge distillation from Qwen2.5-7B-Instruct (frozen 4-bit teacher, response-masked temperature-scaled CE+KL loss) followed by domain-specific supervised fine-tuning (QLoRA, rank 16) on agricultural advisory data.
- **Datasets:** `talhakk/agriculture-qa` (~25.4K rows), `KisanVaani/agriculture-qa-english-only` (~22.6K rows), `AI4Agr/CROP-dataset` (30K rows, capped and English-filtered), plus 1,200 code-generated and arithmetically-verified synthetic examples covering fertilizer-quantity and market-economics calculations.
- **Data quality correction:** diagnosis/pest-category answers were found to omit escalation guidance in early testing; a varied escalation clause was appended to ~70% of these examples (not all, to avoid rigid template memorization), validated against out-of-distribution test prompts.
- **Quantization:** Q4_K_M and Q5_K_M were both built and profiled; **Q4_K_M was selected** based on measured results (see Benchmarks below) — it showed meaningfully higher accuracy and throughput than Q5_K_M, with both comfortably within the memory budget.
- **Domain decision:** Agriculture was selected over an equally-developed Healthcare & Medical candidate after direct comparison. Both domains were built through an identical pipeline; agriculture's outputs were consistently correct and on-persona across test prompts and quant levels, while the healthcare candidate exhibited safety-relevant failures in direct generation testing (repetition-loop degeneration, and one variant recommending an invasive diagnostic procedure in direct violation of its own system instruction not to act as a diagnostic tool) despite targeted data-cleaning efforts.

---

## Constraints

- Target: ADTC Standard Laptop — Intel i5 10th–12th gen / Ryzen 5 3000–5000, 8GB DDR4, integrated graphics only, Ubuntu 22.04.
- Pure CPU inference via llama.cpp — no discrete GPU assumed.
- English-only evaluation, confirmed prior to training; multilingual support was scoped out in favor of deeper single-language domain coverage.
- Context length capped at 2,048 tokens to match the fixed context window used during accuracy evaluation.
- Development and profiling were carried out on a cloud CPU environment (Kaggle) as a proxy for the target laptop hardware; final audit measurements on the reference machine may differ from the figures below.

---

## Benchmarks

Measured via `adtc-profiler run --mode participant`, full run including accuracy (not `--skip-accuracy`).

| Metric | Value |
|---|---|
| Environment | Intel(R) Xeon(R) CPU @ 2.20GHz, 31.3GB RAM, Ubuntu 22.04.5 LTS (cloud CPU proxy) |
| Peak RSS (`memory.peak_rss_mb`) | 1714.29 MB (≈1.67 GB) |
| Steady-state RSS | 1635.33 MB |
| Generation speed (`throughput.tokens_per_second_generation`) | 10.24 tokens/sec |
| First-token latency | 16,473.99 ms (512-token prompt) |
| Accuracy — arc_easy smoke test, 50 samples (`acc_norm`) | 0.70 |
| CPU utilization (p99) | 61.9% |
| Thermal throttling | Not detected (`throttled: false`); core temperature reads `null` — the cloud CPU environment does not expose hardware thermal sensors, so this reflects absence of data rather than a confirmed measurement on physical laptop hardware |
| GGUF parameter count | 1,543,714,304 (matches claimed 1.5B estimate; `params_match: true`) |

**Estimated total score** (using the official formula, `S_perf` capped at 15 TPS reference, `S_eff` against 7GB budget):

S_total ≈ 0.50 × 70 + 0.30 × 68.3 + 0.20 × 76.1 − 0 ≈ **70.7**

This is a self-reported development-time estimate based on the profiler's local accuracy smoke test (`arc_easy`, 50 samples), not the full hidden validation set used in the official audit — it is provided as a directional indicator, not a guaranteed final score.

For comparison, the Q5_K_M variant of this same model measured 9.44 TPS, 1257.74 MB peak RAM, and 0.62 accuracy on the same smoke test — lower on both accuracy and throughput despite its smaller memory footprint, which is why Q4_K_M was selected as the final submission.
