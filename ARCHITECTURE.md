# Whisper JAX Architecture: Pipelines & Online Inference Design

This document provides a comprehensive architectural analysis of the three distinct pipelines used in the `whisper-on-jax` project. It details their specific use cases, design patterns, and the optimization strategies employed for high-performance audio transcription on **TPU v6e**.

## 1. Overview of Pipelines

| Pipeline | File | Primary Use Case | Key Characteristic |
| :--- | :--- | :--- | :--- |
| **Standard Batch** | `pipeline.py` | Offline benchmarking of short audio clips (<30s). | High-throughput batch processing of list inputs. |
| **Long Audio** | `pipeline_long_audio.py` | Offline transcription of long files (e.g., 1 hour+). | **Streaming** chunking with overlap (stride) and timestamp alignment. |
| **Online / Serving** | `pipeline_online.py` | Production API serving (FastAPI). | Asynchronous, non-blocking, hybrid Process/Thread pool architecture. |

---

## 2. Architectural Breakdown

### A. `pipeline.py` (Offline / Short Audio)
Designed for raw throughput benchmarks where all input files are available locally and can be processed in bulk.

*   **Audio Reading:** Uses `ThreadPoolExecutor` to parallelize file reading.
    *   **Standard:** Reads directly using `ffmpeg_read`.
    *   **Speed-Up:** If configured, uses `subprocess` to spawn `ffmpeg` processes that apply `atempo` filters on-the-fly during loading.
*   **Preprocessing:** Simple padding/truncation to 30 seconds.
*   **Batching:** Greedy batching. It iterates through the input list and fills batches up to `batch_size`. Uses **Static Bucketing** (`[4, 40, 80]`) to minimize JAX JIT recompilation.
*   **Decoding:** Uses `jax.pmap` to shard the batch across available TPU devices.

### B. `pipeline_long_audio.py` (Offline / Long Audio)
Designed to handle files that exceed the model's 30-second context window.

*   **Preprocessing (Streaming):** Unlike typical implementations that load 1GB+ WAV files into RAM, this pipeline uses **FFmpeg Streaming**.
    *   It opens a pipe to `ffmpeg` and reads raw float32 bytes in small chunks (30s + stride).
    *   This ensures constant, low memory usage regardless of file size (e.g., 10-hour audio).
*   **Context Management:** Implements a "rolling buffer" logic to handle the overlap (stride) between chunks, ensuring words aren't cut off at boundaries.
*   **Batching:** The unit of work is a "chunk". Chunks from a single file are fed into the batcher.

### C. `pipeline_online.py` (Online API Server)
The most complex and optimized architecture, designed for the "High Throughput, Low Latency" requirements of a production server.

*   **Architecture:** Event-driven, Asynchronous, and Multi-Stage.
*   **Input:** Raw bytes from HTTP requests (FastAPI).
*   **Concurrency:** Handles thousands of simultaneous connections using Python's `asyncio` loop, offloading heavy work to background pools.

---

## 3. Core Components & Strategies

### 1. Audio Reading & Decoding (The GIL Bottleneck)
Python's Global Interpreter Lock (GIL) is the biggest enemy of high-performance serving.
*   **Problem:** `ffmpeg` decoding in a standard thread locks the GIL, freezing the FastAPI server.
*   **Solution (Online Pipeline):** **`ProcessPoolExecutor`**.
    *   We spawn 32 independent *processes* (not threads).
    *   Decoding logic is pickled and sent to these processes.
    *   **Result:** The main server process remains 100% free to handle network traffic. Decoding happens in true parallel across CPU cores.

### 2. Throughput Optimization: Audio Speed-Up
To maximize Token/Second throughput, we employ a "Time-Compression" strategy.
*   **Logic:** Input audio is sped up (e.g., **2.2x**) using FFmpeg's `atempo` filter during the pre-processing stage.
*   **Impact:** A 30-second audio clip becomes ~13.6 seconds. The model processes it faster, effectively boosting the Real-Time Factor (RTFx) by ~40% while maintaining high accuracy.
*   **Implementation:** This is handled transparently in the FFmpeg command generation across all three pipelines.

### 3. Feature Extraction on TPU
*   **Log-Mel Spectrogram:** The raw audio (float array) must be converted into a visual representation (spectrogram) for the model.
*   **CPU Approach:** Calculating this on CPU is slow and consumes significant bandwidth.
*   **TPU Approach (Used Here):** We move raw audio directly to the TPU and compute the spectrogram there using `jax.vmap`.
    *   *Benefit:* Offloads the CPU, allowing it to focus on serving requests and decoding audio.

### 4. Batching Strategy: "The Patient Batcher"
In an online server, requests arrive randomly.
*   **Dynamic "Patient" Batching:**
    *   The batcher thread draws items from the queue.
    *   If the queue is empty, it blocks.
    *   If items are present, it starts a timer (e.g., `0.01s` or `0.2s`).
    *   It accumulates requests until either the **Bucket Size** (80) is reached OR the **Timer Expires**.
    *   *Result:* Under high load, batches are full (maximum efficiency). Under low load, batches are processed immediately (minimum latency).

---

## 4. Online API Inference Design (Detailed Flow)

The `pipeline_online.py` implements a decoupled Producer-Consumer pattern:

1.  **Ingestion (FastAPI Layer):**
    *   User hits `POST /transcribe`.
    *   Server accepts raw bytes (non-blocking).
    *   Creates a `Future` object to track the result.

2.  **Stage 1: Decoding (Process Pool)**
    *   Raw bytes are submitted to the `decode_pool` (ProcessPoolExecutor).
    *   Worker process runs `ffmpeg` to convert bytes -> `float32` numpy array (applying speed-up if configured).
    *   Array is returned to the main process via IPC.

3.  **Stage 2: Queuing**
    *   The decoded array is split into 30s chunks (if necessary).
    *   Chunks are put into the `request_queue`.

4.  **Stage 3: The Batcher (Background Thread)**
    *   Pulls chunks from `request_queue`.
    *   Applies **Padding**: Pads the batch to the nearest bucket size (`[4, 40, 80]`). This is critical because JAX compiles a unique kernel for every shape. Restricting shapes to 3 buckets prevents constant recompilation.
    *   Dispatches batch to TPU via `p_generate` (Parallel Map).

5.  **Stage 4: Inference (TPU Mesh)**
    *   Data is sharded across TPU cores.
    *   Feature extraction runs on TPU.
    *   Transformer Encoder-Decoder runs (Auto-regressive generation).

6.  **Stage 5: Fetching & Post-Processing**
    *   Results sit in TPU memory. A `fetcher_thread` pulls them to CPU asynchronously (hiding PCIE transfer latency).
    *   Tokenizer decodes Token IDs -> Text string.
    *   The original `Future` associated with the request ID is marked as "Done" with the text.

7.  **Response:**
    *   FastAPI sees the `Future` complete and returns the JSON response to the client.

### Optimization Checklist Used
*   ✅ **Priming:** Process pools are forced to spawn workers at startup (dummy jobs) to avoid "cold start" lag.
*   ✅ **JIT Warm-up:** Dummy batches of sizes `[4, 40, 80]` are run through the TPU at startup to compile XLA kernels beforehand.
*   ✅ **Asyncio + Threading + Multiprocessing:** Using all three concurrency models where they shine best (IO, lightweight logic, heavy CPU work).