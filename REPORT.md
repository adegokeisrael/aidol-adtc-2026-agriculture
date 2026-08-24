# Technical Report — AIDOL Agricultural Advisory Assistant

**Team ID:** AgriMind: Offline AI Model for African Farmers
**Submitter:** Adegoke Israel Adedolapo
**Domain:** Agriculture — Crop, Pest, Input-Cost, and Market Advisory
**Model:** AIDOL-Agri-Advisor-1.5B-Q4_K_M
**Base Model:** Qwen2.5-1.5B-Instruct
**Runtime:** llama.cpp (GGUF, CPU-only)
**Date:** August 2026

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement and Motivation](#1-problem-statement-and-motivation)
3. [Domain Selection](#2-domain-selection)
4. [Datasets](#3-datasets)
5. [Model Selection](#4-model-selection)
6. [Methodology](#5-methodology)
7. [Quantization](#6-quantization)
8. [Evaluation and Profiling](#7-evaluation-and-profiling)
9. [Comparative Domain Validation: Agriculture vs. Healthcare](#8-comparative-domain-validation-agriculture-vs-healthcare)
10. [Engineering Challenges and Solutions](#9-engineering-challenges-and-solutions)
11. [Final Submission Summary](#10-final-submission-summary)
12. [Limitations and Future Work](#11-limitations-and-future-work)
13. [Conclusion](#12-conclusion)

---

## Executive Summary

This report documents the end-to-end development of an offline, CPU-only agricultural advisory assistant built for the Africa Deep Tech Challenge (ADTC) 2026. The system is a 1.5-billion-parameter language model, produced by distilling a larger 7-billion-parameter teacher model and then fine-tuning the resulting student on domain-specific agricultural data, before quantizing it to the GGUF format for efficient inference on commodity laptop hardware (8GB RAM, integrated graphics, no GPU). The final submitted model answers self-contained, text-only questions across four sub-tasks within the agriculture domain: crop symptom diagnosis, fertilizer/input calculation, pest and disease management, and market/storage economics. Every design decision documented below — domain choice, model size, distillation strategy, dataset composition, and quantization level — was made against the ADTC's published scoring formula:

```
S_total = 0.50 · S_acc + 0.30 · S_perf + 0.20 · S_eff − P_thermal
```
(plus up to +10 for demonstrated African-context relevance)

Two quantization variants (Q4_K_M and Q5_K_M) were built, profiled, and directly compared using the official `adtc-profiler`. **Q4_K_M was selected as the final submission** based on measured results: it showed higher accuracy (0.70 vs. 0.62) and higher throughput (10.24 vs. 9.44 tokens/sec) than Q5_K_M, with both variants comfortably within the 7GB memory budget. This is a data-driven reversal of an earlier, pre-profiling assumption that the higher-fidelity Q5_K_M would outperform — the measured numbers, not the assumption, determined the final choice.

---

## 1. Problem Statement and Motivation

Small-scale and informal farming remains the economic backbone of much of Nigeria and the wider West African region, yet farmers and extension workers in these communities frequently operate with limited, delayed, or unreliable access to agronomic expertise — particularly in rural areas with poor connectivity. A cloud-dependent AI assistant is of little use in these conditions: it requires a stable internet connection, ongoing data costs, and server infrastructure that may not reach the communities that need it most.

The objective of this project was therefore to build a genuinely offline advisory tool — a language model small and efficient enough to run entirely on a low-specification laptop (the ADTC "Standard Laptop" reference: Intel i5 10th–12th gen or AMD Ryzen 5 3000–5000 series, 8GB DDR4 RAM, integrated graphics only, zero cloud dependency during evaluation) — while still being accurate and useful enough to meaningfully assist a farmer or extension officer.

The system is **stateless by design**: each prompt is self-contained, with all necessary context (quantities, symptoms, region, prices) embedded directly in the question. This matches both realistic field usage and the structure of the competition's own accuracy evaluation, which tests the raw model against standalone prompts rather than a multi-turn conversation.

This constraint — real accuracy inside an 8GB RAM ceiling, on CPU only — is the central engineering tension this report resolves, and it shaped every subsequent decision described below.

---

## 2. Domain Selection

The Africa Deep Tech Challenge offers seven candidate domains. Before committing engineering time to any one of them, each was evaluated against three practical criteria: **(a)** how well the task fits a stateless, single-turn question-and-answer format, since the official accuracy evaluation is prompt-based, not conversational; **(b)** the depth and quality of publicly available training data; and **(c)** the risk profile of a small, quantized model being wrong.

### 2.1 Domain Comparison

| Domain | Stateless Q&A fit | Public dataset depth | Small-model risk | Notes |
|---|---|---|---|---|
| Healthcare / Medical | Good | Strong (exam-style) | High — clinical harm risk, hallucination-prone | Exam-style datasets mismatch a patient-education register; safety-critical failure mode |
| **Agriculture** | **Excellent** | **Strong (forum + structured)** | **Medium — advisory tone is more forgiving** | **Selected — see 2.2** |
| Coding | Excellent | Abundant | High — binary correctness, harsh under quantization | Crowded submission lane |
| Creative Writing | Fair | Abundant | Low technical risk, but 'accuracy' scoring is ill-defined | Judging criteria ambiguous for subjective quality |
| Corporate / Enterprise | Good (with embedded-data prompts) | Moderate | Medium | Viable alternative, considered and set aside |
| Autonomous AI Agents | Poor — implies state/orchestration | Sparse | Medium | Mismatched to a stateless, single-model evaluation |

### 2.2 Why Agriculture

Agriculture was selected for four concrete reasons:

- **Dataset availability**: multiple public, purpose-built agricultural Q&A corpora exist (`talhakk/agriculture-qa`, `KisanVaani/agriculture-qa-english-only`, `AI4Agr/CROP-dataset`), providing both exam-adjacent and — critically — naturalistic, forum-sourced question phrasing close to how a real farmer would ask.
- **Lower liability risk than healthcare**: an imperfect agronomic recommendation is a materially lower-stakes failure than an imperfect clinical one, which matters when the deployed system is a quantized 1.5B model that will occasionally be wrong.
- **A built-in African-context narrative**: the CGIAR GARDIAN corpus explicitly frames its content around advisory services for smallholder producers in low- and middle-income countries — a direct, evidence-backed match for the challenge's African Use Case bonus criteria, rather than a bonus justification invented after the fact.
- **Empirical confirmation**: a direct head-to-head comparison against a parallel healthcare build (documented in Section 8) showed agriculture's outputs were consistently coherent and correctly reasoned, while the healthcare variant exhibited repetition-loop failures and, more seriously, recommended a diagnostic procedure — a direct violation of the safety framing given to that model.

The domain was deliberately **narrowed rather than treated broadly**: instead of a general "agricultural assistant," the model was scoped to four concrete sub-tasks — crop symptom diagnosis, fertilizer/input calculation, pest and disease management, and market/storage economics — because a small model achieves near-ceiling accuracy on a narrow, well-defined task far more reliably than it does on an open-ended one.

---

## 3. Datasets

### 3.1 Sources Used

| Dataset | Content | Rows used | Role in training mix |
|---|---|---|---|
| `talhakk/agriculture-qa` | General farming + 10 priority-crop deep-dive Q&A | ~25,400 | Crop diagnosis, general agronomy |
| `KisanVaani/agriculture-qa-english-only` | Real farmer-forum Q&A, English-only | ~22,600 | Naturalistic question phrasing |
| `AI4Agr/CROP-dataset` | Rice/maize-focused Alpaca-format QA (bilingual EN/ZH) | 30,000 (English-filtered, capped) | Crop-specific reasoning depth |
| Synthetic — fertilizer calculation | Code-generated, arithmetic verified in software (not by an LLM) | ~600 | Guaranteed-correct numeric reasoning |
| Synthetic — market/storage economics | Code-generated, arithmetic verified in software | ~600 | Guaranteed-correct comparative reasoning |
| `CGIAR/gardian-cigi-ai-documents` | 65K smallholder-advisory research documents | Reference only | Grounding source for teacher-generated answers; African-context justification |

### 3.2 Data Quality Interventions

Three corrections were applied to the raw sources before training, each in response to a defect surfaced during iterative spot-checking:

- **Language filtering**: `AI4Agr/CROP-dataset` silently mixes English and Chinese examples with no language column. A CJK-character regex filter was added before capping the sample, since the evaluation is English-only and unfiltered sampling risked polluting the training set with untranslated Chinese text.
- **Escalation-note augmentation**: initial spot-checks showed the model never mentioned when a farmer should seek in-person expert help, because neither `talhakk` nor `KisanVaani`'s raw answers include this. A post-processing pass appended one of four varied escalation phrasings to ~70% of diagnosis/pest examples (not 100%, to avoid teaching a robotic, verbatim-repeated clause).
- **Arithmetic ground-truthing**: for the two numeric sub-tasks (fertilizer calculation, market economics), example generation and answer computation were both done in deterministic code rather than by an LLM, so every training label is guaranteed arithmetically correct — removing the need for a separate verification/filtering pass on this bucket.

### 3.3 Held-Out Evaluation Data

A validation subset was reserved from the training split (5%) for loss-based evaluation during fine-tuning. Separately, a hand-written set of out-of-distribution spot-check prompts — using numeric values and scenario framings deliberately absent from the synthetic generators' sampling ranges (e.g. hectare counts, storage costs, and price combinations outside the generators' discrete option lists, including at least one case engineered so that the "expected" pattern from training reverses) — was used to distinguish genuine numeric generalization from memorization of the synthetic bucket.

---

## 4. Model Selection

### 4.1 Base Architecture: Qwen2.5-1.5B-Instruct

Qwen2.5-1.5B-Instruct was selected as the base/student architecture for four reasons:

- **Apache 2.0 license** — no restriction on public redistribution of the fine-tuned weights, a hard requirement since the competition mandates publicly hosted GGUF weights.
- **Strong instruction-following and general reasoning** for its parameter class, which matters for whatever portion of the automated accuracy evaluation may be domain-general rather than agriculture-specific.
- **Mature, well-tested GGUF/llama.cpp conversion support** — the competition's runtime is llama.cpp + GGUF exclusively, so conversion reliability was treated as a hard constraint, not a nice-to-have; a broken or unsupported conversion path is an automatic disqualification, not merely a low score.
- **A size point chosen deliberately in the middle of the feasible range**: small enough to leave comfortable headroom under the 7GB effective RAM budget at 4–5 bit quantization, but large enough to retain meaningfully more domain knowledge than a 0.5B-class model — a real consideration given accuracy is weighted 2.5× more heavily than efficiency in the scoring formula (50% vs. 20%).

### 4.2 Teacher Model: Qwen2.5-7B-Instruct

The teacher model was chosen from the same model family as the student specifically to avoid cross-tokenizer alignment problems during distillation (see Section 5.2) — both models share the same underlying vocabulary, which makes direct logit-level knowledge transfer tractable without a token-remapping step.

---

## 5. Methodology

### 5.1 Pipeline Overview

```
Teacher Qwen2.5-7B-Instruct → Knowledge Distillation → Student Qwen2.5-1.5B (base) → LoRA/QLoRA SFT → Merged FP16 Model
      → GGUF Conversion (convert_hf_to_gguf.py) → Quantization (Q4_K_M / Q5_K_M) → ADTC Profiler → Final Submission
```

The pipeline proceeds in five stages: **(1)** distill a compact student model from a larger teacher on general-reasoning data; **(2)** supervised fine-tune the distilled student on the agriculture-specific dataset described in Section 3; **(3)** merge the resulting LoRA adapter into the base weights; **(4)** convert to GGUF and quantize; and **(5)** validate every claimed metric against the official ADTC profiler before submission.

### 5.2 Why Knowledge Distillation — and Why Before Fine-Tuning

An initial direct supervised fine-tune (SFT) of the plain Qwen2.5-1.5B-Instruct base was run first as a baseline, specifically to test whether distillation was necessary at all before investing the additional time it requires. This baseline run informed the decision to add a distillation stage ahead of domain fine-tuning, for the following reasons:

- A small model fine-tuned narrowly on a single domain risks losing general reasoning ability it started with — a known failure mode sometimes called catastrophic forgetting. Distilling from a stronger 7B teacher first, on broader reasoning-oriented data, gives the 1.5B student a stronger and more robust starting point before domain-specific narrowing begins.
- The competition's accuracy evaluation has a genuinely ambiguous automated component: organizer documentation describes it as a mix of judge-scored domain prompts and an automated benchmark, while the grading tool's own default configuration runs a generic reasoning benchmark (ARC-Easy) unless overridden. Distillation on broad reasoning data is a hedge against this ambiguity — it improves the student's general-reasoning floor regardless of which interpretation turns out to be correct, without costing anything on the domain-specific side.
- Distillation transfers not just final answers but the teacher's output distribution (its relative confidence across all possible next tokens), which is a richer training signal per example than raw text labels alone — valuable when the compute and data budget for training a 1.5B model from scratch on reasoning tasks is limited.

#### 5.2.1 Distillation Method: Logit-Based (Hinton-style) Knowledge Distillation

The specific technique used is classical response-based knowledge distillation via a temperature-scaled Kullback-Leibler (KL) divergence loss between the teacher's and student's output probability distributions, combined with a standard cross-entropy (CE) loss against the ground-truth labels:

```
Loss = α · CE(student, labels) + (1 − α) · T² · KL(softmax(student_logits / T) ‖ softmax(teacher_logits / T))
```

with α = 0.5 and temperature T = 2.0 as starting hyperparameters. This approach — rather than feature-level or sequence-level distillation — was chosen because it required no architectural changes to either model, works directly on causal-LM logits without an auxiliary alignment step, and is the best-established, most reproducible distillation method in the literature, which matters given the tight competition timeline left little room for an unproven or exploratory technique.

Two implementation details were necessary to make this work in practice:

- **Loss masking**: the KL and CE losses were computed only over the assistant's response tokens, not the system/user portion of each training example, so the student was never rewarded or penalized for tokens it does not actually generate at inference time.
- **Vocabulary-padding alignment**: although teacher and student share the same real tokenizer vocabulary, each model independently pads its output (`lm_head`) layer to a different hardware-friendly size internally (152,064 vs. 151,936 logits). The two logit tensors were sliced to their shared, smaller dimension before computing the KL divergence — otherwise the loss computation fails on a shape mismatch, since the extra padding positions carry no real token meaning.

#### 5.2.2 Memory Management for Dual-Model Training

Running a 7B teacher and a 1.5B student simultaneously on a single 16GB training GPU required: loading the teacher in 4-bit precision, frozen and inference-only (no gradients, no optimizer state); loading the student in 4-bit precision with a trainable LoRA adapter; capping sequence length below the level used for plain SFT to leave memory headroom for the additional teacher forward pass required at every training step; and explicitly freeing the teacher from memory before reloading a full-precision copy of the student for the LoRA-merge step, to avoid both models briefly coexisting at full precision.

### 5.3 Supervised Fine-Tuning (SFT)

Following distillation, the student model was further fine-tuned on the agriculture dataset (Section 3) using LoRA (Low-Rank Adaptation) via the Hugging Face `peft` and `trl` libraries, with 4-bit (NF4) quantized base weights during training (QLoRA). LoRA adapters were applied to all attention and MLP projection layers, rank 16. This two-stage design — broad distillation first, narrow domain SFT second — reflects a deliberate separation of concerns: distillation builds general reasoning capacity; SFT specializes that capacity for the target task, without needing to relearn general language ability from scratch.

The LoRA adapter was subsequently merged into the base weights (`merge_and_unload`) to produce a single, standalone model suitable for GGUF conversion — the competition's evaluation pipeline requires a single self-contained weights file, not a base model plus a separate adapter.

---

## 6. Quantization

### 6.1 Why Quantization Is Necessary

The competition enforces a hard 8GB RAM ceiling on the reference laptop hardware, with no GPU acceleration available. A 1.5B-parameter model stored in its native 16-bit floating point format occupies roughly 3GB of memory before any inference overhead (KV-cache, context buffer) is added — quantization to lower-precision integer representations is what makes CPU-only inference within this budget tractable, and is standard practice for on-device LLM deployment.

### 6.2 Method: GGUF / llama.cpp, K-Quant Variants

The merged FP16 model was converted to the GGUF format using llama.cpp's `convert_hf_to_gguf.py`, then quantized using `llama-quantize` into two candidate variants for evaluation:

| Variant | Approx. bits/weight | Approx. file size |
|---|---|---|
| Q4_K_M | ~4.5 bits (mixed) | ~986 MB |
| Q5_K_M | ~5.5 bits (mixed) | ~1.13 GB |

GGUF and llama.cpp were non-negotiable choices, not preferences: the competition's submission rules mandate llama.cpp + GGUF as the only accepted runtime, and its automated grading tool parses the GGUF file's own tensor header directly to cross-check the declared parameter count against the actual file — making an incorrect or non-standard format an automatic validation failure rather than a scoring penalty.

### 6.3 Selecting the Final Quantization Level

Both quantization levels were empirically profiled (Section 7) rather than chosen by assumption. Because the scoring formula's throughput term is capped once a fixed reference speed is exceeded, while the efficiency term has no such ceiling, the deciding factor between Q4_K_M and Q5_K_M was measured, real-world performance across all three scored dimensions — not an assumption that higher bit-depth necessarily means a better outcome.

**Result: Q4_K_M was selected.** The measured comparison (full figures in Section 7.3) showed Q4_K_M outperforming Q5_K_M on *both* accuracy (0.70 vs. 0.62) and throughput (10.24 vs. 9.44 tokens/sec), while both remained comfortably within the memory budget. This runs counter to a naive assumption that higher-fidelity quantization should always score better — it did not, on this model and this task, and the final decision followed the data rather than the assumption.

---

## 7. Evaluation and Profiling

### 7.1 The ADTC Profiler

The official `adtc-profiler` tool was used to generate reproducible, standardized measurements for every candidate model variant, run in CPU-only mode (no GPU offload) to approximate the reference hardware. The tool measures four categories directly against the scoring formula:

| Metric | How it is measured | Weight |
|---|---|---|
| S_acc (Accuracy) | Automated benchmark score + judge-scored responses to 2 submitted + 2 hidden domain prompts | 50% |
| S_perf (Throughput) | `llama-bench` generation speed, capped at a fixed reference TPS | 30% |
| S_eff (Efficiency) | Peak RSS memory sampled during generation, relative to a 7GB budget | 20% |
| P_thermal (Thermal) | Flat penalty if peak CPU temperature reaches a throttling threshold | −10 (flat) |

### 7.2 Methodology Note on Score Interpretation

A close reading of the profiler's own source code revealed two findings that directly shaped engineering priorities: first, the throughput score is capped at 100% once a fixed reference tokens-per-second value (15.0) is exceeded — it is not scored relative to the fastest competing submission, meaning speed beyond a safe margin above that threshold earns no further points. Second, the efficiency score has no equivalent ceiling — every additional megabyte of RAM saved continues to contribute proportionally. Given this, and given accuracy's 50% weight is 2.5× the efficiency term's 20%, the optimization priority adopted was: clear the throughput cap with a safe margin, then allocate all further effort to accuracy, with memory efficiency treated as a secondary but still-scored concern.

Note also that **P_thermal is measured on the organizers' own audit sandbox**, not from a participant's self-reported profiler output — confirmed directly by the organizers during the pre-submission Q&A session. Self-reported thermal figures below are therefore informational only, not the value that will be scored.

### 7.3 Results

Measured via `adtc-profiler run --mode participant`, full run including accuracy (not `--skip-accuracy`), on a cloud CPU environment (Kaggle) as a proxy for the target laptop hardware.

**Environment:** Intel(R) Xeon(R) CPU @ 2.20GHz, 31.3GB RAM, Ubuntu 22.04.5 LTS (cloud CPU proxy — final audit measurements on the reference machine may differ from the figures below).

| Metric | Q4_K_M (submitted) | Q5_K_M (evaluated, not submitted) |
|---|---|---|
| Generation speed (`throughput.tokens_per_second_generation`) | **10.24 tok/sec** | 9.44 tok/sec |
| First-token latency (512-token prompt) | 16,473.99 ms | 32361.65 |
| Peak RSS (`memory.peak_rss_mb`) | 1714.29 MB (≈1.67 GB) | 1257.74 MB (≈1.23 GB) |
| Steady-state RSS | 1635.33 MB | 1179.07 |
| Accuracy — `arc_easy` smoke test, 50 samples (`acc_norm`) | **0.70** | 0.62 |
| CPU utilization (p99) | 61.9% | 70.1 |
| Thermal throttling | Not detected (`throttled: false`); `core_temp_c_peak` reads `null` — cloud CPU environment does not expose hardware thermal sensors, so this reflects absence of data rather than a confirmed measurement on physical laptop hardware | — |
| GGUF parameter count | 1,543,714,304 (matches claimed 1.5B estimate; `params_match: true`) |  1,543,714,304 (matches claimed 1.5B estimate; `params_match: true`) |
| Est. S_perf | 68.3 | 62.9 |
| Est. S_eff | 76.1 | 82.5 |
| **Est. S_total** | **≈ 70.7** | ≈ 66.4 |

**Estimated total score calculation** (Q4_K_M, official formula, S_perf capped at 15 TPS reference, S_eff against 7GB budget):

```
S_total ≈ 0.50 × 70 + 0.30 × 68.3 + 0.20 × 76.1 − 0 ≈ 70.7
```

This is a self-reported development-time estimate based on the profiler's local accuracy smoke test (`arc_easy`, 50 samples), **not** the full hidden validation set used in the official audit — it is provided as a directional indicator, not a guaranteed final score.

Despite Q5_K_M's smaller memory footprint (and correspondingly higher S_eff), its lower accuracy and lower throughput produced a lower estimated total score, which is why **Q4_K_M was selected as the final submission.**

### 7.4 Qualitative Spot-Checks

Beyond the automated profiler, both quantization levels were manually tested against the two domain test prompts and a set of out-of-distribution prompts not present in the synthetic generators' sampling space, to verify genuine numeric reasoning rather than memorization. Both variants produced coherent, on-topic responses across the crop-diagnosis and fertilizer-calculation test categories; Q4_K_M's measured advantage on both the automated accuracy benchmark and throughput — combined with comparable qualitative output quality — was the deciding factor in its selection over Q5_K_M.

---

## 8. Comparative Domain Validation: Agriculture vs. Healthcare

To validate the domain choice empirically rather than by assumption alone, a parallel pipeline (identical distillation → SFT → quantization → profiling process) was built for a healthcare/medical variant, using the same base and teacher models, for direct comparison.

### 8.1 Findings

- **Register mismatch**: an initial healthcare model trained on exam-style question banks (MedMCQA, MedQA-USMLE) reproduced board-exam phrasing, including a fabricated textbook citation attached to an unrelated topic — a direct consequence of training on multiple-choice exam data rather than patient-facing dialogue.
- **Dataset correction**: switching to a real doctor-patient dialogue dataset (MedDialog) improved tone and naturalism substantially, but surfaced a safety-relevant regression: the model recommended a specific drug dosage, directly contradicting its own system-prompt instruction to defer dosing questions to a clinician.
- **Residual risk after mitigation**: even after adding dosage- and diagnostic-language filters to the training data, the higher-fidelity Q5_K_M healthcare variant still recommended a specific diagnostic procedure (a lumbar puncture) in response to a triage-education prompt — a direct violation of its "not a diagnostic tool" framing, and evidence that filtering alone could not fully suppress behavior already present in the base model's pretrained medical knowledge.

### 8.2 Conclusion of the Comparison

This side-by-side result — agriculture consistently coherent and correctly reasoned; healthcare exhibiting both generation-quality failures (repetition loops) and a genuine safety-framing violation — confirmed the domain-selection reasoning in Section 2 empirically rather than theoretically, and is the direct basis for submitting the agriculture variant as the final entry.

---

## 9. Engineering Challenges and Solutions

| Challenge | Root cause | Resolution |
|---|---|---|
| Dataset schema mismatches | Assumed column names did not match actual Hugging Face dataset schemas (e.g. `answer` vs. `answers`) | Print-and-inspect step added before every dataset mapping step, catching mismatches before silent data loss |
| Missing dataset split | `AI4Agr/CROP-dataset` ships only a `test` split, not `train` | Loaded the available split directly; verified this was a naming artifact, not a genuine held-out benchmark split |
| Cross-lingual contamination | `AI4Agr/CROP-dataset` silently mixes English and Chinese examples with no language column | CJK-character regex filter applied prior to sampling |
| Teacher/student vocabulary mismatch | Same-family models pad their output layer to different sizes internally (152,064 vs. 151,936) | Logit tensors sliced to the shared dimension before computing KL-divergence loss |
| Environment/dependency drift | Library version mismatches across `trl`, `transformers`, and `torchao` caused repeated training-time errors | Explicit version pinning; re-asserted pins after any step (e.g. llama.cpp's own `requirements.txt`) that could silently reinstall a newer version |
| Exposed API credential | A Hugging Face access token was inadvertently hardcoded in notebook source | Token revoked and regenerated; migrated to Kaggle's encrypted Secrets manager, removing all hardcoded credentials from source |
| Profiler model-path resolution failure | The grading tool reads the model file location from a required `_runtime.model_path` field in `metadata.json`, which had been omitted | Field restored per the official submission template documentation; verified against a successful profiler run |
| Quantization assumption reversal | Initial assumption favored Q5_K_M for better fidelity | Measured profiler results showed Q4_K_M outperforming on both accuracy and throughput; final selection followed the data |

---

## 10. Final Submission Summary

| Component | Detail |
|---|---|
| Domain | Agriculture — crop diagnosis, fertilizer calculation, pest/disease management, market economics |
| Base / student model | Qwen2.5-1.5B-Instruct (Apache 2.0) |
| Teacher model | Qwen2.5-7B-Instruct |
| Training method | Logit-based knowledge distillation, followed by LoRA/QLoRA supervised fine-tuning |
| Quantization | **GGUF, Q4_K_M (submitted)** — selected over Q5_K_M based on measured accuracy and throughput |
| Runtime | llama.cpp, CPU-only inference, zero cloud dependency |
| Weights hosting | Publicly hosted on Hugging Face; downloaded fresh via `download_model.sh` |
| African Use Case relevance | Nigerian-context smallholder farming scenarios (Naira pricing, local units — bags, cartons, hectares), grounded against the CGIAR smallholder-advisory corpus |

---

## 11. Limitations and Future Work

- The automated portion of the accuracy evaluation's exact composition (fully domain-specific vs. partly generic reasoning) could not be confirmed with certainty from public documentation alone; the distillation stage was included partly as a hedge against this ambiguity rather than a fully confirmed requirement.
- Training data drawn from public forum and dataset sources (`talhakk`, `KisanVaani`) has not been independently verified against a formal agronomic reference for every example; a future iteration would benefit from expert review of a representative sample, mirroring the verification applied to the numeric synthetic buckets.
- Profiling was conducted on a cloud CPU environment (Kaggle) as a proxy for the target laptop hardware; a Docker memory-capped run was used as the closest available approximation to the audit sandbox, but true reference-hardware numbers — and the organizer-measured thermal figures in particular — may differ from the self-reported figures in Section 7.3.
- The current system is text-only and single-turn by design, matching the evaluation format; a richer multi-turn, retrieval-augmented product layer (e.g. ingesting a farmer's own sales or field records) was scoped as an application-layer enhancement outside the benchmarked model itself, since the accuracy evaluation tests the raw model directly and would not reflect any such retrieval layer.

---

## 12. Conclusion

This project delivered a fully offline, CPU-only agricultural advisory model, built through a deliberate, evidence-driven pipeline: an empirically validated domain choice, a same-family teacher-student distillation stage to establish a strong general-reasoning foundation, targeted supervised fine-tuning on a carefully cleaned and augmented agriculture dataset, and a quantization strategy chosen through direct profiling against the competition's actual scoring mechanics rather than assumption. Every major design decision in this report — from domain selection through final quantization level — was tested against real data, real profiler output, or a direct comparison against an alternative approach, rather than taken on faith.
