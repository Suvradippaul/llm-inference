# LLM Inference Performance Analysis: The Roofline Model in Practice

> *A hands-on study of throughput limits, latency behaviour, and the physics of GPU inference.*

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Prerequisites & Mental Model](#2-prerequisites--mental-model)
3. [Hardware & Model Overview](#3-hardware--model-overview)
4. [The Roofline Model — Theory](#4-the-roofline-model--theory)
5. [LLM Inference: Prefill vs Decode](#5-llm-inference-prefill-vs-decode)
6. [Benchmarking Setup & Configuration](#6-benchmarking-setup--configuration)
7. [Test 1: Max Throughput — Rate Sweep](#7-test-1-max-throughput--rate-sweep)
8. [Test 2: Long Context — Prefill Stress](#8-test-2-long-context--prefill-stress)
9. [Test 3: High Concurrency — Scheduler Stress](#9-test-3-high-concurrency--scheduler-stress)
10. [Synthesis: All Limits Together](#10-synthesis-all-limits-together)
11. [Optimization Levers](#11-optimization-levers)
12. [Quick Decision Guide for Practitioners](#12-quick-decision-guide-for-practitioners)
13. [Glossary](#13-glossary)

---

## 1. Introduction

When deploying a large language model at scale, two questions always surface:

- **How many requests per second can this hardware serve?**
- **At what point does adding more traffic increase latency without increasing throughput?**

These are not tuning questions — they are physics questions. The answer is determined by the hardware's memory bandwidth and compute throughput, the model's size and architecture, and how inference workloads map onto those resources.

This document studies those limits empirically. We use the **Roofline Model**, a classical performance-analysis framework from HPC, adapted here for LLM serving. While the benchmarks in this document run a specific model (Mistral Small 24B) on specific hardware (NVIDIA H100), **the analysis is generic**. You can plug any dense transformer's parameters and any GPU's specifications into the formulas below to predict its ceilings.

**What you will learn:**

- Why LLM decode is almost always *memory-bandwidth-bound*, not compute-bound
- How to predict the throughput ceiling from first principles *before* running a benchmark
- Why latency explodes at high request rates even when throughput stays flat
- What the KV cache costs in memory bandwidth and HBM capacity
- Which optimizations actually move the ceiling — and which do not

---

## 2. Prerequisites & Mental Model

> [!NOTE]
> **Prerequisites:** Basic familiarity with GPU concepts (what a forward pass is, what HBM is) will help. Unfamiliar terms are defined in the [Glossary](#13-glossary) and linked inline where possible.

Sure. Starting with the first one:

---

**Section 2.1 — The Library Analogy**

---

### 2.1 The Library Analogy

Before diving into the math, it helps to build an intuition for how GPUs process LLMs.

Think of the GPU like a gigantic reference library:
- **The Tensor Cores (Compute)** are the reading desks where librarians actively process information.
- **The High-Bandwidth Memory or HBM (Storage)** holds the massive collection of books (the model weights).

When a request comes in to generate a word, a librarian must reference the *entire collection of books* — reading all the model weights into the cores. The limitation is almost never how fast the librarian can read at the desk (compute capacity). The bottleneck is **how fast the books can be carried from the shelves to the desk** (memory bandwidth).

This is the dominant intuition for **decode** — the phase where the model generates one token at a time, loading the full weight collection on every single step.

However, the analogy has a limit. When processing a very long prompt — say, 16,000 tokens — the librarian suddenly has an enormous amount of material to work through all at once. At that point, the shelves are no longer the bottleneck: **the desk fills up first**. The librarian can fetch books faster than they can be read. This is the compute-bound regime, and it is what governs the prefill phase at long contexts.

The shift between these two regimes — shelves bound vs desk bound — is precisely what the Roofline Model quantifies. We will build that framework in Section 4.

---

## 3. Hardware & Model Overview

To ground our mathematical examples, we use the following hardware and model configurations throughout the document.

### 3.1 GPU Example

| Property | Value |
|---|---|
| GPU | NVIDIA H100 SXM5 |
| HBM3 Capacity | 80 GB |
| Memory Bandwidth | 3.35 TB/s |
| FP16/BF16 Peak Compute | ~989 TFLOPS |
| Serving Framework | vLLM (v0.10.2) |

The two numbers that matter most for inference limit prediction are the absolute max limits: **memory bandwidth** (3.35 TB/s) and **peak compute** (989 TFLOPS).

### 3.2 Model Example

| Property | Value |
|---|---|
| Model | Mistral Small 3.2 24B Instruct |
| Precision | BF16 |
| Parameters | ~24 Billion |
| Weight size in memory | ~48 GB |
| Number of layers | 40 |
| Hidden dimension | 5120 |
| Attention heads | 32 (Q) / 8 (KV, [GQA](#13-glossary)) |
| Head dimension | 128 |

At BF16 precision (2 bytes/parameter), 24B parameters occupy **~48 GB** of HBM simply resting on the shelves — leaving ~32 GB of RAM available for the [KV Cache](#13-glossary) headroom. 

---

## 4. The Roofline Model — Theory

The **Roofline Model** is a performance framework that determines whether a workload is bottlenecked by the GPU's "shelves" (memory bandwidth) or its "desks" (compute). 

### 4.1 Arithmetic Intensity

To figure out which bottleneck applies, we calculate a metric called **Arithmetic Intensity ($I$)**:

$$I = \frac{\text{FLOPs performed}}{\text{Bytes transferred from memory}} \quad \text{[FLOP/byte]}$$

How much math do you do for every byte of data you load?
- High intensity → Compute-bound (desks are full)
- Low intensity → Memory-bandwidth-bound (waiting on shelves)

### 4.2 Building the Roofline

On any hardware, performance is capped by two absolute physics ceilings:

1. **The compute roof:** Peak TFLOPS of the GPU.
2. **The bandwidth slope:** How fast you can feed data to the compute units.

```
Peak FLOPS ─────────────────────────────────────── (ceiling 1: compute roof)
               ╱
              ╱  ← slope = Peak BW
             ╱     (ceiling 2: bandwidth slope)
────────────╱──────────────────────────────────── I (FLOP/byte)
```

The mathematical formula for performance is:
$$\text{Attainable Performance} = \min\!\left(I \times \text{Peak BW},\;\text{Peak Compute}\right)$$

### 4.3 The Ridge Point

The **ridge point** is the exact crossover intensity where the workload transitions from being memory-bound to compute-bound. It's simply the ratio of the hardware's two peaks:

$$I_{\text{ridge}} = \frac{\text{Peak Compute}}{\text{Peak BW}}$$

For an H100 GPU:
$$I_{\text{ridge}} = \frac{989 \times 10^{12}\ \text{FLOP/s}}{3.35 \times 10^{12}\ \text{Bytes/s}} \approx 295 \;\text{FLOP/byte}$$

For any kernel with $I < 295\;\text{FLOP/byte}$, the GPU is memory-bandwidth-bound: adding faster compute units does nothing.

> [!TIP]
> **Checkpoint:** Arithmetic intensity tells you whether adding more compute (FLOPS) or more memory bandwidth helps. Higher intensity → compute matters. Lower intensity → bandwidth matters. For the H100, that tipping point is 295 FLOP/byte.

---

## 5. LLM Inference: Prefill vs Decode

Every inference request passes through two fundamentally different phases. 

**Section 5.1 — Prefill Phase**

---

### 5.1 Prefill Phase

All input tokens are processed in **a single parallel forward pass**.
- Time scales with input length (ISL): more tokens → more FLOPs
- Parallelism is high — the full sequence is processed at once
- Output: the KV cache for all input tokens, and the first output token

For a single request with ISL tokens processed in parallel, the arithmetic intensity is approximately:

$$I_{\text{prefill}} \approx \frac{2 \times N_{\text{params}} \times \text{ISL}}{N_{\text{params}} \times 2} = \text{ISL} \;\text{FLOP/byte}$$

- **Numerator** ($2 \times N_{\text{params}} \times \text{ISL}$): A matrix-vector multiply of the form $y = Wx$ costs roughly $2 \times n_{\text{weights}}$ FLOPs. Since ISL tokens are processed in parallel, total FLOPs scale linearly with ISL.
- **Denominator** ($N_{\text{params}} \times 2$): All model weights must be loaded from HBM once. At BF16 precision, each parameter occupies 2 bytes, so byte traffic is $N_{\text{params}} \times 2$.
- The $N_{\text{params}}$ and the factor of 2 cancel, leaving intensity = ISL.

*(Note: This approximation ignores the memory traffic of writing the KV cache to HBM, which is small relative to the ~48 GB of model weights at typical sequence lengths.)*

At ISL = 512:

$$I_{\text{prefill}} = 512 \;\text{FLOP/byte} > 295 \;\text{FLOP/byte (H100 ridge point)}$$

**Prefill at ISL = 512 is compute-bound.** And because intensity scales directly with ISL, longer prompts only push further past the ridge — at ISL = 16K, intensity reaches 16,384 FLOP/byte, which is 55× above the ridge. We will see exactly what that means for throughput in Test 2.

**Why single-request prefill?** vLLM's chunked prefill pipeline processes one prefill sequence at a time, interleaving its chunks with ongoing decode steps rather than prefilling multiple sequences simultaneously. This means the single-request analysis is the operationally correct one — even under heavy load, prefill intensity per sequence is ISL, not ISL × B. The analysis would only generalise to ISL × B if multiple sequences were prefilled in parallel, which this configuration does not do.

> **Checkpoint:** Prefill is compute-bound from relatively short contexts onward, and becomes increasingly so at longer ones. This is the opposite of decode, where the bottleneck is memory bandwidth.

---

### 5.2 Decode Phase

Output tokens are generated **one at a time**, autoregressively. Each step:
- Processes 1 new token per sequence in the batch
- Loads the *entire* set of model weights from HBM (48 GB)
- Loads the KV cache for the full context so far
- Outputs exactly 1 new token per sequence

For a batch of size $B$, the intensity is approximately:

$$I_{\text{decode}} \approx \frac{2 \times N_{\text{params}} \times B}{N_{\text{params}} \times 2} = B \;\text{FLOP/byte}$$

- **Numerator** ($2 \times N_{\text{params}} \times B$): Each step still applies the full weight matrices to *every* sequence in the batch simultaneously, so total FLOPs scale with batch size B.
- **Denominator** ($N_{\text{params}} \times 2$): Regardless of how many sequences are batched, the weights are loaded from HBM only *once* per step (the GPU broadcasts the same weights across the batch). Byte cost stays fixed at $N_{\text{params}} \times 2$.
- This is the key insight: **batching is free from a memory-bandwidth perspective** — you get B× more FLOPs from the same weight read, which is why intensity = B.

*(Note: Again, this is a weight-only approximation that ignores KV cache read traffic, which we rigorously factor in later. Adding KV cache reads slightly lowers intensity further).*

| Batch size B | Arithmetic intensity | Regime |
|---|---|---|
| 1 | ~1 FLOP/byte | Severely bandwidth-bound |
| 100 | ~100 FLOP/byte | Still bandwidth-bound |
| **295** | **~295 FLOP/byte** | **Ridge point (balanced)** |
| 500 | ~500 FLOP/byte | Compute-bound |

For typical serving batch sizes of $B = 50$–$150$, decode sits deep in the bandwidth-bound regime. 

> [!NOTE]
> **Ceiling 1 Introduced (Bandwidth):** Decode throughput scales linearly with batch size, capped entirely by how fast weights matrix stream from HBM (the **bandwidth ceiling**), not by tensor cores.

### 5.3 Which Phase Dominates?

For typical workloads, decode runs hundreds of times per request while prefill runs once. **Decode dominates total runtime**.

*(Exception: At very long contexts, e.g., ISL=16K, prefill requires massive FLOPs and becomes the dominant per-request bottleneck).*

> [!TIP]
> **Checkpoint:** Prefill (reading the prompt) is usually fast and compute-bound. Decode (typing the answer) is slow and memory-bandwidth-bound.

---

## 6. Benchmarking Setup & Configuration

We use `bench_serving.py` (from SGLang) to send HTTP requests to a running vLLM server, simulating real-world workloads.

### 6.1 Server Configuration (vLLM 0.10.2)

To understand exactly what is happening in the engine during tests, inspect the vLLM environment configuration:

- **`dtype=bfloat16`**: Weights are strictly stored in BF16 format (2 bytes per param).
- **`tensor_parallel_size=1`**: Single GPU execution.
- **`chunked_prefill_enabled=True`**: Massive prefills are automatically chunked and interleaved with decode steps to prevent outright head-of-line blocking. 
- **`enable_prefix_caching=True`**: Identical system prompt prefixes are automatically cached in HBM.
- **`max_capture_size=512`** (CUDA Graphs): vLLM compiles CUDA graphs for all batch sizes up to 512. This removes CPU dispatch overhead, ensuring that our measurements are purely testing GPU hardware limits, not Python overhead.

### 6.2 Key Benchmarking Parameters

| Parameter | Description |
|---|---|
| `--request-rate` | Simulated Poisson arrival rate (req/s). `inf` = all sent immediately |
| `--random-input-len` | Target input token count per request |
| `--max-concurrency` | Cap on simultaneous in-flight requests in the server |

### 6.3 Three Tests, Three Ceilings

We stress the system to reveal the three underlying physical ceilings of LLM serving:

| Test | What it stresses | Revealed Ceiling |
|---|---|---|
| **#1: Max Throughput** | Request rate sweep | **Bandwidth Ceiling** (HBM Speed) |
| **#2: Long Context** | Input length sweep | **Compute Ceiling** (during prefill) |
| **#3: High Concurrency** | Max concurrent requests | **Capacity Ceiling** (HBM Size) |

---

## 7. Test 1: Max Throughput — Rate Sweep

**Setup:** ISL = 512, OSL = 512, request rate swept from 5 req/s to ∞. We want to find the maximum possible throughput the hardware can serve before latency balloons.

---

### 7.1 Prediction: The Upper Bound Before We Measure

The Roofline model lets us establish one firm prediction before running a single request.

Each decode step must stream the full 48 GB of model weights from HBM regardless of batch size:

$$t_{\text{step}}^{\min} = \frac{48 \times 10^9}{3.35 \times 10^{12}} \approx 14.3 \;\text{ms}$$

If KV cache reads were free, this is the fastest any decode step could run. At a steady-state batch of B concurrent requests each needing 512 decode steps:

$$\text{Throughput}_{\max} = \frac{B}{\text{OSL} \times t_{\text{step}}} = \frac{B}{512 \times 0.0143}$$

This is our prior upper bound — the ceiling assuming zero KV bandwidth cost. The actual throughput will land somewhere below it, depending on how much HBM bandwidth the KV cache consumes on top of weight streaming.

---

### 7.2 Results

| Rate (req/s) | Actual Tput (req/s) | Mean E2E (ms) | P99 E2E (ms) | Mean TTFT (ms) | Mean TPOT (ms) |
|---|---|---|---|---|---|
| 5 | 4.77 | 4,033 | 17,621 | 57.5 | 21.1 |
| 10 | 9.12 | 4,855 | 20,907 | 76.2 | 25.2 |
| 20 | **15.54** | 9,211 | 44,426 | 157.8 | 50.1 |
| 50 | 16.40 | 34,197 | 76,283 | 18,760 | 120.4 |
| ∞ | 16.21 | 57,250 | 109,757 | 41,600 | 129.9 |

Throughput saturates hard at **~16 req/s**. No amount of additional traffic moves it.

---

### 7.3 Retroactive Verification: Tightening the Bounds

With results in hand, we can now compute the effective batch size and use it to verify the observed ceiling against both a lower and upper Roofline bound.

**Finding the effective batch size via Little's Law:**

At the saturation point (rate = 20, throughput = 15.54 req/s, mean E2E = 9.21 s):

$$B_{\text{eff}} = \lambda \times W = 15.54 \times 9.21 \approx 143 \;\text{requests}$$

**Upper bound — weights only, no KV cost (prior prediction):**

$$\text{Throughput}_{\max} = \frac{143}{512 \times 0.0143} \approx 19.5 \;\text{req/s}$$

**Lower bound — weights plus full KV cost (retroactive):**

Each token costs 160 KB of KV reads (40 layers × 8 KV heads × 128 head dim × 2 for K and V × 2 bytes). At mid-decode the average context per sequence is ISL + OSL/2 = 768 tokens. With B = 143:

$$\text{KV}_{\text{step}} = 143 \times 768 \times 160\;\text{KB} \approx 17.5 \;\text{GB}$$

$$t_{\text{step}}^{\max} = \frac{48 + 17.5}{3.35 \times 10^3} \approx 19.5 \;\text{ms}$$

$$\text{Throughput}_{\min} = \frac{143}{512 \times 0.0195} \approx 14.3 \;\text{req/s}$$

**The verdict:**

$$\boxed{14.3 \;\text{req/s} \leq \text{Observed: } 15.54 \;\text{req/s} \leq 19.5 \;\text{req/s}}$$

The observed plateau sits squarely between the two bounds. Memory bandwidth is the dominant bottleneck. The KV cache imposes a real but partially mitigated cost on top of weight streaming — vLLM's PagedAttention is neither perfectly efficient nor completely wasteful with KV bandwidth.

---

### 7.4 Why Latency Explodes Past Saturation

Throughput and latency tell different stories at high request rates, and it is worth understanding why they diverge.

At rate = 20, throughput is 15.54 req/s and mean E2E is 9.2 seconds. At rate = 50, throughput is still 16.4 req/s — essentially unchanged — but P99 latency has jumped to 76 seconds. Pushing to infinite rate keeps throughput flat while P99 reaches 110 seconds.

This is a queuing phenomenon, not a GPU one. The server has a fixed maximum service rate of ~16 req/s. Once the arrival rate exceeds that, incoming requests queue before the GPU touches them. By Little's Law, as the queue grows, waiting time grows with it — without bound. The GPU is not slowing down; requests are simply spending longer waiting in line.

**The practical implication:** a server operating at 100% of its throughput ceiling is not in a stable state. Any traffic burst tips the queue into unbounded growth. Target **75–80% of the measured ceiling** in production to keep latency predictable.

---

## 8. Test 2: Long Context — Prefill Stress

**Setup:** Concurrency capped at 32, ISL swept from 1K to 16K tokens, infinite traffic to saturate the server. Goal: understand how long prompts stress the system beyond the bandwidth ceiling established in Test 1.

---

### 8.1 Prediction: Two Ceilings Hit Simultaneously

At long contexts, two independent problems compound each other. We can predict both before running.

**Compute contention during prefill:**

From Section 5.1, prefill arithmetic intensity is approximately ISL FLOP/byte. At ISL = 16K:

$$I_{\text{prefill}} = 16,384\text{ FLOP/byte} \gg 295\text{ FLOP/byte} \text{ (H100 ridge)}$$

Prefill is deeply compute-bound — 55× past the ridge point. Even with chunked prefill interleaving execution, the total FLOPs required do not decrease. Every prefill chunk competes for tensor core time with ongoing decode work, and at this intensity, prefill wins that competition heavily.

**HBM capacity exhaustion:**

Each token requires 160 KB of KV cache. With 32 concurrent sequences at ISL = 16K tokens each, the KV memory required is:

$$\text{KV}_{\text{required}} = 32 \times 16{,}384 \times 160\;\text{KB} \approx 81.9\;\text{GB}$$

This already exceeds the entire 80 GB HBM — before a single output token is generated. We therefore predict that at ISL = 16K, the server will breach physical memory capacity and be forced into KV swapping, where blocks are evicted to CPU memory and recomputed on demand. This will cause throughput to collapse independently of the compute contention problem.

**Prior prediction:** at ISL = 16K with concurrency = 32, both the compute ceiling and the capacity ceiling are breached simultaneously. Throughput should collapse severely and TTFT should reach several seconds.

---

### 8.2 Results

| Input Length | Mean TTFT (ms) | Req Tput (req/s) | P99 E2E (ms) |
|---|---|---|---|
| 1,024 | 112 | 9.01 | 6,830 |
| 8,192 | 818 | 2.51 | 26,744 |
| **16,384** | **11,053** | **1.26** | **42,854** |

TTFT reaches 11 seconds at ISL = 16K. Throughput collapses from 9.01 req/s at ISL = 1K to 1.26 req/s — an 86% drop.

Both predicted failures materialise. TTFT scaling confirms compute contention: the GPU is spending increasingly large chunks of time on prefill FLOPs, starving decode. The throughput collapse beyond what compute contention alone would cause confirms KV swapping: vLLM is evicting KV blocks to CPU memory and recomputing them, destroying generation speed.

> **Ceilings 2 and 3 encountered simultaneously.** Long contexts are dangerous not because either problem alone is fatal, but because they compound. Compute contention slows everything down while memory exhaustion forces expensive recomputation on top.

---

## 9. Test 3: High Concurrency — Scheduler Stress

**Setup:** ISL = 1024, OSL = 1024, rate = ∞, `--max-concurrency` swept from 50 to 1000. Goal: find the concurrency level at which adding more in-flight requests stops helping and starts hurting.

---

### 9.1 Prediction: The Capacity Ceiling

We know from Test 1 that throughput is capped at ~16 req/s by memory bandwidth. The question here is different: how many requests can the GPU hold concurrently before it physically runs out of space for their KV caches?

After loading model weights, available HBM is:

$$\text{Available for KV} = 80\;\text{GB} - 48\;\text{GB} = 32\;\text{GB}$$

At 160 KB per token, this fits a maximum of:

$$N_{\text{max tokens}} = \frac{32 \times 10^9}{163{,}840} \approx 195{,}000\;\text{tokens}$$

At ISL + OSL = 2048 tokens per sequence, the absolute concurrency limit before HBM is exhausted is:

$$B_{\text{max}} = \frac{195{,}000}{2{,}048} \approx 95\;\text{sequences}$$

**Prior prediction:** throughput should plateau around concurrency = 95–100. Beyond that, the GPU has no room for additional KV caches. New requests will queue rather than execute, and latency will grow without bound while throughput stays flat.

---

### 9.2 Results

| Max Concurrency | Req Tput (req/s) | Mean TTFT (ms) | Mean TPOT (ms) |
|---|---|---|---|
| 50 | 9.67 | 88.7 | 26.1 |
| 100 | **14.18** | 155.3 | 35.8 |
| 200 | 16.40 | 403.6 | 60.5 |
| 1,000 | 16.22 | 25,307 | 129.0 |

Throughput plateaus between concurrency = 100 and 200, consistent with our predicted capacity limit of ~95 sequences. Beyond that point, the server accepts requests but immediately queues them — throughput gains nothing while TTFT balloons from 155 ms to 25 seconds. The GPU is not getting more work done; it is simply accumulating a longer waiting line.

---

**Section 10 — Synthesis: All Limits Together**

---

## 10. Synthesis: All Limits Together

The three tests reveal three independent physical ceilings, each governing a different dimension of serving performance. But they are not fully independent — the ceilings interact, and understanding those interactions is what lets you reason about optimizations clearly.

**The three ceilings:**

**Ceiling 1 — Memory Bandwidth:** How fast model weights stream from HBM per decode step. This sets the maximum throughput plateau. No matter how many requests arrive, the server cannot exceed ~16 req/s because every decode step must load 48 GB of weights through a 3.35 TB/s pipe. Batch size determines how much useful work happens per weight load, but the pipe width is fixed.

**Ceiling 2 — Compute:** How many FLOPs the tensor cores can execute per second. This ceiling is mostly invisible at short contexts — prefill at ISL = 512 is fast enough not to disrupt decode. It becomes dominant at long contexts, where a single prefill at ISL = 16K requires 55× more arithmetic intensity than the ridge point, consuming tensor core time that would otherwise serve decode steps.

**Ceiling 3 — HBM Capacity:** The physical memory available for KV caches after weights are loaded. At 32 GB available and 160 KB per token, the GPU can hold approximately 195,000 tokens in flight. This caps maximum concurrency to ~95 sequences at 2K context length.

**How the ceilings interact:**

The important insight is that raising one ceiling often moves another. Consider quantization: moving from BF16 to INT4 shrinks model weights from 48 GB to 12 GB. This directly raises Ceiling 1 — fewer bytes to stream per decode step means higher throughput. But it simultaneously raises Ceiling 3 — the 36 GB freed from weights becomes available for KV cache, pushing maximum concurrency from ~95 sequences to roughly 220. One optimization, two ceilings raised.

Tensor parallelism works similarly: splitting weights across N GPUs multiplies both aggregate memory bandwidth (Ceiling 1) and total HBM capacity (Ceiling 3) by N. The cost is inter-GPU communication overhead, which grows with N and eventually becomes its own bottleneck.

Ceiling 2 is the exception — it does not benefit from quantization or tensor parallelism in the same way. The total FLOPs required to prefill a 16K token sequence are fixed by the model architecture, not by weight precision or GPU count. Chunked prefill is the primary lever for Ceiling 2, and even that only distributes the work more fairly — it does not reduce it.

**The practical picture:**

For most production workloads with moderate context lengths, Ceiling 1 is the binding constraint. Ceiling 3 becomes relevant when serving many concurrent long-context requests. Ceiling 2 surfaces only at very long prompts or in RAG pipelines where input lengths regularly exceed 8K tokens. Knowing which ceiling you are against determines which optimization is worth pursuing — a decision guide for exactly this is in Section 12.

---

**Section 11 — Optimization Levers**

---

## 11. Optimization Levers

Each optimization below targets a specific ceiling. The first question to ask before applying any of them is: which ceiling are you actually against? Applying the wrong lever does nothing — quantizing weights when you are hitting Ceiling 2 during long-context prefill will not help, because the bottleneck is FLOPs, not bytes.

---

### 11.1 Quantization → Targets Ceiling 1 and Ceiling 3

Quantization shrinks model weights, reducing the bytes streamed per decode step and freeing HBM for KV cache.

**Ceiling 1 impact — throughput:**

The throughput ceiling scales inversely with weight size:

$$\text{Throughput ceiling} \propto \frac{\text{Peak BW}}{\text{Weight bytes per step}}$$

For our Mistral 24B example, the weight-only step time at BF16 is 14.3 ms, giving an upper bound of ~19.5 req/s. At INT4 (12 GB weights):

$$t_{\text{step}}^{\min} = \frac{12 \times 10^9}{3.35 \times 10^{12}} \approx 3.6 \;\text{ms}$$

The throughput upper bound shifts from ~19.5 req/s toward ~78 req/s — a 4× improvement in the ceiling, before KV bandwidth tax.

**Ceiling 3 impact — concurrency:**

Freed HBM goes directly to KV cache. Moving from BF16 to INT4 frees 36 GB, expanding available KV memory from 32 GB to 68 GB. Maximum concurrency at 2K context grows from ~95 sequences to approximately:

$$B_{\text{max}} = \frac{68 \times 10^9}{163{,}840} \approx 415{,}000 \;\text{tokens} \div 2{,}048 \approx 200 \;\text{sequences}$$

**Caveat:** Quantization introduces accuracy degradation, which varies by method (INT8, FP8, INT4) and model. Always validate output quality on your specific workload before deploying quantized weights.

---

### 11.2 Tensor Parallelism → Targets Ceiling 1 and Ceiling 3

Increasing `tensor_parallel_size` to N splits weights across N GPUs. Each GPU reads only 1/N of the weights per step, multiplying effective bandwidth and total HBM capacity by N.

**Ceiling 1 impact:** With 2× H100s in tensor parallel, aggregate bandwidth doubles to 6.7 TB/s. The weight-only step time halves, and the throughput ceiling approximately doubles.

**Ceiling 3 impact:** Total HBM doubles from 80 GB to 160 GB. Available KV memory grows from 32 GB to 112 GB (160 GB minus the same 48 GB of weights, now split but still totalling the same parameter count). Maximum concurrency at 2K context grows from ~95 to approximately 340 sequences.

**Caveat:** Inter-GPU communication overhead grows with N. Each decode step requires synchronisation across GPUs via NVLink. At high N, this communication cost becomes its own bottleneck and linear scaling degrades. For a single 24B model, tensor parallel beyond 2–4 GPUs typically yields diminishing returns.

---

### 11.3 Chunked Prefill → Targets Ceiling 2

Chunked prefill breaks a large compute-bound prefill into smaller chunks, interleaving them with decode steps rather than blocking the GPU for the full duration of the prefill.

**What it does and does not do:** It does not reduce the total FLOPs required to process a long prompt — a 16K token prefill requires the same arithmetic regardless of chunking. What it does is prevent any single prefill from monopolising the tensor cores for its entire duration, allowing decode steps to proceed between chunks. This reduces TTFT variance and prevents decode throughput from collapsing completely during long prefills.

**When it matters:** Chunked prefill has negligible impact at short contexts (ISL ≤ 1K) where prefill completes quickly anyway. Its benefit scales with ISL — at ISL = 16K, without chunking, ongoing decode sequences would be fully blocked for the entire prefill duration.

**Ceiling 2 is the only ceiling chunked prefill addresses.** It does not improve throughput under Ceiling 1 conditions and does not expand KV capacity.

---

### 11.4 Speculative Decoding → Targets Ceiling 1

Speculative decoding uses a small draft model to propose K candidate tokens, which the main model verifies in a single forward pass. When the acceptance rate is high, this generates multiple tokens per main-model forward pass — effectively reducing the number of full weight loads required per output token.

The expected speedup is:

$$\text{Speedup} = \frac{1 + K \times \alpha}{1 + \text{drafting overhead}}$$

where $\alpha$ is the token acceptance rate.

**Ceiling 1 impact:** If $\alpha$ is high (e.g. 0.8) and K = 4, each main model step produces on average 4.2 tokens instead of 1 — reducing the effective number of 48 GB weight loads per output token by roughly 4×. The throughput ceiling rises proportionally.

**Caveat:** The draft model itself consumes HBM and compute. Acceptance rate $\alpha$ is highly sensitive to the match between draft model and main model distributions — it degrades on creative or diverse outputs and improves on structured or repetitive ones. Measure $\alpha$ on your specific workload before assuming the theoretical speedup.

---

**Section 12 — Quick Decision Guide for Practitioners**

---

## 12. Quick Decision Guide for Practitioners

The three tests in this document map directly to three failure modes you will encounter in production. When you observe a symptom, the goal is to identify which ceiling you are against before reaching for an optimization. Applying the wrong fix wastes engineering time and sometimes makes things worse.

---

### Identifying Your Ceiling

**Step 1:** Measure throughput and latency simultaneously under your production load pattern. Do not measure them separately — the relationship between them is the signal.

**Step 2:** Apply the decision logic below.

---

| Symptom | Ceiling | Diagnosis | Recommended Action |
|---|---|---|---|
| Throughput plateaus despite increasing traffic. Latency climbs steeply past the plateau. | Ceiling 1 — Bandwidth | Memory bandwidth is saturated. Decode steps are bottlenecked by weight streaming. | Quantize weights (INT8 or INT4). Add GPUs via tensor parallelism. Cap incoming rate to 75–80% of measured plateau to stabilise latency. |
| Throughput is low and TTFT is very high. TPOT for other concurrent requests also degrades during long prompts. | Ceiling 2 — Compute | Long-context prefill is consuming tensor core time needed for decode. | Verify chunked prefill is enabled. If already enabled, reduce `max_seq_len` to limit prefill intensity, or implement disaggregated prefill to isolate prefill compute from decode. |
| "CUDA out of memory" errors, or TPOT degrades sharply as concurrency increases past a threshold. | Ceiling 3 — Capacity | KV cache memory is exhausted. vLLM is swapping KV blocks to CPU and recomputing. | Reduce max concurrency. Enable prefix caching if requests share common system prompts. Quantize weights to free HBM for KV cache. Add GPUs to increase total HBM. |
| All three metrics look healthy at low load but degrade together under burst traffic. | All three — Compounding | You are operating close to multiple ceilings simultaneously. Bursts tip you over all of them at once. | This is an architectural problem, not a tuning one. The serving infrastructure needs more capacity — additional GPUs, a larger model parallelism configuration, or request rate limiting upstream. |

---

### A Note on Operating Point

Every ceiling in this document is a hard physical limit, not a tuning target. The practical operating point for a stable production system is **75–80% of whichever ceiling is binding**. At 100% of the ceiling, any traffic burst — even a brief one — tips the system into unbounded queue growth. At 75–80%, the system has enough headroom to absorb transient spikes without latency spiralling.

If your workload regularly operates above 80% of the bandwidth ceiling, that is not a sign to tune more aggressively — it is a signal that the infrastructure needs to scale.

---

## 13. Glossary

- **ISL / OSL:** Input/Output Sequence Length (number of tokens in prompt/generation).
- **TTFT (Time To First Token):** Latency from request submission to the first output token.
- **TPOT (Time Per Output Token):** Mean per-step decode latency as experienced by a single request.
- **E2E latency:** End-to-end latency from request submission to the final generated token.
- **GQA (Grouped Query Attention):** Multiple query heads share one KV head pair, significantly reducing KV cache memory footprint.
- **KV Cache:** Stores attention intermediate states for each token, preventing recalculation of past tokens.
- **PagedAttention:** vLLM's memory manager that allocates KV cache in fixed-size blocks to eliminate fragmentation.
- **Prefix Caching:** Reusing precomputed KV cache blocks across requests that share an identical system prompt.
