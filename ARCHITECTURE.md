# Whisper JAX Architecture: Pipelines & Online Inference Design

This document provides a comprehensive architectural analysis of the three distinct pipelines used in the `whisper-on-jax` project. It details their specific use cases, design patterns, and the optimization strategies employed for high-performance audio transcription on TPU v5e.

## 1. Overview of Pipelines

| Pipeline | File | Primary Use Case | Key Characteristic |
| :--- | :--- | :--- | :--- |
| **Standard Batch** | `pipeline.py` | Offline benchmarking of short audio clips (<30s). | High-throughput batch processing of list inputs. |
| **Long Audio** | `pipeline_long_audio.py` | Offline transcription of long files (e.g., 1 hour+). | chunking with overlap (stride) and timestamp alignment. |
| **Online / Serving** | `pipeline_online.py` | Production API serving (FastAPI). | Asynchronous, non-blocking, hybrid Process/Thread pool architecture. |

---

## 2. Architectural Breakdown

### A. `pipeline.py` (Offline / Short Audio)
Designed for raw throughput benchmarks where all input files are available locally and can be processed in bulk.

*   **Audio Reading:** Uses `ffmpeg` via a `ThreadPoolExecutor` to read files from disk. This is "blocking" I/O but parallelized across threads.
*   **Preprocessing:** Simple padding/truncation to 30 seconds.
*   **Batching:** Greedy batching. It iterates through the input list and fills batches up to `batch_size`. Uses **Static Bucketing** (`[4, 40, 80]`) to minimize JAX JIT recompilation.
*   **Decoding:** Uses `jax.pmap` to shard the batch across available TPU devices.

### B. `pipeline_long_audio.py` (Offline / Long Audio)
Designed to handle files that exceed the model's 30-second context window.

*   **Preprocessing (Chunking):** The input file is decoded entirely into memory and then sliced into 30s segments with a configurable **stride** (e.g., 5s overlap).
    *   *Stride:* Ensures context isn't lost at the boundaries of chunks.
*   **Batching:** The unit of work becomes a "chunk" rather than a "file". A single long file produces many batch items.
*   **Post-Processing:** Requires re-assembling the transcribed text from chunks. (Note: For simpler throughput benchmarks, this pipeline often counts "samples processed" rather than doing complex timestamp merging).

### C. `pipeline_online.py` (Online API Server)
The most complex and optimized architecture, designed for the "High Throughput, Low Latency" requirements of a production server.

*   **Architecture:** Event-driven, Asynchronous, and Multi-Stage.
*   **Input:** Raw bytes from HTTP requests (FastAPI).
*   **Concurrency:** Handles thousands of simultaneous connections using Python's `asyncio` loop, offloading heavy work to background pools.

---

## 3. Core Components & Strategies

### 1. Audio Reading & Decoding (The GIL Bottleneck)
Python's Global Interpreter Lock (GIL) is the biggest enemy of high-performance serving.
*   **Problem:** `ffmpeg` decoding in a standard thread locks the GIL, freezing the FastAPI server. If 50 users send audio, the server stops accepting new connections while decoding the first few.
*   **Solution (Online Pipeline):** **`ProcessPoolExecutor`**.
    *   We spawn 32 independent *processes* (not threads).
    *   Decoding logic is pickled and sent to these processes.
    *   **Result:** The main server process remains 100% free to handle network traffic. Decoding happens in true parallel across CPU cores.

### 2. Pre-processing & Feature Extraction
*   **Log-Mel Spectrogram:** The raw audio (float array) must be converted into a visual representation (spectrogram) for the model.
*   **CPU Approach:** calculating this on CPU is slow and consumes significant bandwidth.
*   **TPU Approach (Used Here):** We move raw audio directly to the TPU and compute the spectrogram there using `jax.vmap`.
    *   *Benefit:* Offloads the CPU, allowing it to focus on serving requests and decoding audio.

### 3. Batching Strategy: "The Patient Batcher"
In an online server, requests arrive randomly.
*   **Naive Approach:** Process requests immediately as they arrive. -> *Low Latency, Terrible Throughput (1 item/batch).*
*   **Strict Batching:** Wait until we have 80 items. -> *Great Throughput, Terrible Latency (first user waits forever).*
*   **Our Solution:** **Dynamic "Patient" Batching**.
    *   The batcher thread draws items from the queue.
    *   If the queue is empty, it blocks.
    *   If items are present, it starts a timer (e.g., `0.01s`).
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
    *   Worker process runs `ffmpeg` to convert bytes -> `float32` numpy array.
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
