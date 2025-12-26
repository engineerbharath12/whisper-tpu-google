
import argparse
import os
import time
import numpy as np
import jax
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from transformers.utils import logging

# Local Imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from whisper_jax.pipeline_online import FlaxWhisperOnlinePipeline

console = Console()
logging.set_verbosity_error()

# Configuration
AUDIO_BASE_PATH = "/home/brathinam_google_com/whisper/26dec/whisper-tpu-google/asr_audio_new"
SHORT_AUDIO_FILE = os.path.join(AUDIO_BASE_PATH, "18s", "medical_domain_test.wav")

def run_benchmark(pipeline, concurrency):
    console.print(Panel(f"[bold blue]Test: E2E Latency Benchmark with Concurrency: {concurrency}[/bold blue]", expand=False))

    if not os.path.exists(SHORT_AUDIO_FILE):
        console.print(f"[bold red]❌ ERROR: Audio file not found at {SHORT_AUDIO_FILE}.[/bold red]")
        return

    # Load audio once
    with open(SHORT_AUDIO_FILE, "rb") as f:
        audio_bytes = f.read()

    console.print(f"--- 🚀 Submitting {concurrency} concurrent requests... ---")
    
    futures = []
    start_times = []
    
    global_start_time = time.time()
    
    for _ in range(concurrency):
        start_times.append(time.time())
        futures.append(pipeline(audio_bytes, task="transcribe", return_timestamps=False))

    results = []
    latencies = []
    
    for i, future in enumerate(futures):
        try:
            res = future.result()
            end_time = time.time()
            latencies.append(end_time - start_times[i])
            results.append(res)
        except Exception as e:
            console.print(f"[red]Request {i} failed: {e}[/red]")

    global_end_time = time.time()
    total_wall_time = global_end_time - global_start_time
    
    if not latencies:
        console.print("[red]No successful requests.[/red]")
        return

    avg_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p99_latency = np.percentile(latencies, 99)
    
    # RTFx Calculation
    # RTFx = (Total Audio Duration Processed) / (Total Wall Time)
    # Total Audio Duration = 18s * concurrency
    # This is "Throughput RTFx"
    
    # Per-request RTFx = 18s / Latency
    
    # We will report Throughput RTFx as it's more standard for batch processing
    total_audio_duration = 18.0 * concurrency # Approx 18s
    throughput_rtfx = total_audio_duration / total_wall_time
    
    table = Table(title=f"E2E Latency & Performance (Concurrency {concurrency})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Total Wall Time", f"{total_wall_time:.4f} s")
    table.add_row("Avg Latency/Req", f"{avg_latency:.4f} s")
    table.add_row("P50 Latency", f"{p50_latency:.4f} s")
    table.add_row("P99 Latency", f"{p99_latency:.4f} s")
    table.add_row("Throughput (RTFx)", f"{throughput_rtfx:.2f} x")
    table.add_row("Throughput (Req/s)", f"{concurrency / total_wall_time:.2f} req/s")
    
    console.print(table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="openai/whisper-large-v3-turbo")
    args = parser.parse_args()

    # JAX Cache
    try:
        from jax import config
        JAX_CACHE_DIR = "/home/brathinam_google_com/whisper/26dec"
        os.makedirs(JAX_CACHE_DIR, exist_ok=True)
        config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)
        config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        config.update("jax_persistent_cache_min_compile_time_secs", 0)
    except:
        pass

    console.print(Panel(f"[bold green]Initializing Pipeline for {args.model_id}[/bold green]", expand=False))
    
    pipeline = FlaxWhisperOnlinePipeline(args.model_id, dtype=jax.numpy.bfloat16)

    # Concurrency levels to test
    concurrencies = [1, 80, 200]
    
    for c in concurrencies:
        run_benchmark(pipeline, c)
        time.sleep(2) # Cooldown

