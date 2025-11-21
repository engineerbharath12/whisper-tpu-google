# Whisper JAX on TPU v5e: Benchmark Results Comparison

This document summarizes the performance benchmarks comparing **Whisper Large V3** (Standard) and **Whisper Large V3 Turbo** across different usage scenarios.

## System Configuration
*   **Hardware:** Google TPU v5e (Single Node)
*   **Precision:** `bfloat16`
*   **Framework:** JAX + Flax
*   **Feature Extraction:** TPU-accelerated

---

## 1. Summary Comparison Table

| Metric | Whisper Large V3 (Standard) | Whisper Large V3 **Turbo** | Improvement |
| :--- | :--- | :--- | :--- |
| **Max Throughput (Online)** | ~1,779x | **~3,601x** | **2.02x** |
| **Max Throughput (Offline)** | ~2,072x | **~4,007x** | **1.93x** |
| **Latency (1280 Reqs)** | 13.01s | **6.43s** | **2x Faster** |
| **Model Architecture** | 32 Decoder Layers | 4 Decoder Layers | 8x Smaller Decoder |

---

## 2. Detailed Benchmark Data

### Scenario A: Online API Server (High Concurrency)
*   **Pipeline:** `pipeline_online.py` (FastAPI + Async ProcessPool + Patient Batcher)
*   **Audio:** 18s (`medical_domain_test.wav`)
*   **Concurrency:** 1280 simultaneous requests

| Model | Throughput (RTFx) | Total Time (1280 Reqs) | Pipeline Efficiency (Internal) |
| :--- | :--- | :--- | :--- |
| **Whisper Large V3** | 1,779x | 13.01 s | 3.04x |
| **Whisper Large V3 Turbo** | **3,601x** | **6.43 s** | **6.87x** |

> **Observation:** The Turbo model effectively doubles the serving capacity of the API server, clearing the queue of 1280 requests in half the time.

### Scenario B: Offline Batch Processing (Short Audio)
*   **Pipeline:** `pipeline.py` / `libri-short.py` (ThreadPool Decoding + Greedy Batching)
*   **Audio:** 18s (`medical_domain_test.wav`)
*   **Batch Size:** 80

| Model | Concurrency | Throughput (RTFx) | Total Time |
| :--- | :--- | :--- | :--- |
| **Whisper Large V3** | 1280 | 2,072x | 11.33 s |
| **Whisper Large V3 Turbo** | 1280 | **4,007x** | **5.86 s** |

> **Observation:** Offline processing removes the network/API overhead, showing the raw theoretical limit of the TPU pipeline. Turbo breaks the **4000x** barrier.

### Scenario C: Offline Long-Audio Processing
*   **Pipeline:** `pipeline_long_audio.py` / `libri-long.py` (Chunking + Stride)
*   **Audio:** 5 minutes (`videoplayback_5min.wav`)
*   **Total Audio Processed:** 50 minutes (10 concurrent files)

| Model | Throughput (RTFx) | Total Time |
| :--- | :--- | :--- |
| **Whisper Large V3** | 1,654x | 1.81 s |
| **Whisper Large V3 Turbo** | **2,818x** | **1.06 s** |

> **Observation:** Even with the overhead of chunking and striding long files, Turbo maintains a massive lead (~1.7x).

---

## 3. Conclusion
For high-volume production workloads on TPU v5e:

1.  **Performance:** **Whisper Large V3 Turbo** is vastly superior, offering **~2x throughput** across all scenarios.
2.  **Cost Efficiency:** By doubling the throughput per chip, Turbo effectively **halves the serving cost** per hour of audio processed.
3.  **Quality:** Transcription accuracy on the test samples was identical between V3 and Turbo.
