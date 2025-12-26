import argparse
import time
import os
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import jax.numpy as jnp

# ===== Local/Project Imports =====
from whisper_jax.pipeline_online import FlaxWhisperOnlinePipeline
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
console = Console()

# --- Configuration ---
# Updated to match the environment
AUDIO_BASE_PATH = "/home/brathinam_google_com/whisper/26dec/whisper-tpu-google/asr_audio_new"
WARMUP_FILE = os.path.join(AUDIO_BASE_PATH, "18s", "medical_domain_test.wav")

# --- Test Scenarios ---
CONCURRENT_BENCHMARK_SCENARIOS = {
    18: [1, 80, 200],
}

SHORT_AUDIO_FILES = {
    2: os.path.join(AUDIO_BASE_PATH, "2s", "good_day.wav"),
    8: os.path.join(AUDIO_BASE_PATH, "8s", "462_gu.wav"),
    14: os.path.join(AUDIO_BASE_PATH, "14s", "hindi_15_sec.wav"),
    18: os.path.join(AUDIO_BASE_PATH, "18s", "medical_domain_test.wav"),
}

def main():
    parser = argparse.ArgumentParser(description="Benchmark E2E RTFx and latency distribution for Online Pipeline.")
    parser.add_argument("--model_id", type=str, default="openai/whisper-large-v3-turbo", help="Hugging Face model ID.")
    parser.add_argument("--batch_size", type=int, default=80, help="Max batch size for the pipeline (buckets will be adjusted).")
    args = parser.parse_args()

    # ===== JAX CACHE CONFIGURATION =====
    try:
        from jax import config
        import warnings
        JAX_CACHE_DIR = "/whisper/jax-cache"
        os.makedirs(JAX_CACHE_DIR, exist_ok=True)
        config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)
        console.print(f"--- JAX Cache enabled. Using directory: {JAX_CACHE_DIR} ---")
    except ImportError:
        warnings.warn("Could not configure JAX cache.")

    # Instantiate Online Pipeline
    console.print(Panel(f"[bold green]Instantiating Online Pipeline: {args.model_id}[/bold green]", expand=False))
    
    # Note: OnlinePipeline reads batch_buckets from config.yml, but we can verify/log them.
    # It doesn't take batch_size as init arg directly in the same way, usually relies on config buckets.
    # But let's instantiate it.
    pipeline = FlaxWhisperOnlinePipeline(
        checkpoint=args.model_id, 
        dtype=jnp.bfloat16
    )

    # Warmup
    # Online pipeline typically has its own warmup routine in __init__, but let's send a dummy request to be sure.
    console.print(f"\n--- Performing Extra Warm-up ---")
    if os.path.exists(WARMUP_FILE):
        with open(WARMUP_FILE, "rb") as f:
            warmup_bytes = f.read()
        
        # Send enough requests to trigger a batch execution
        warmup_futures = []
        # We assume batch buckets include something small like 4.
        for _ in range(4):
            warmup_futures.append(pipeline(warmup_bytes))
        
        for future in warmup_futures:
            try:
                future.result()
            except Exception as e:
                console.print(f"[bold red]    -> Warm-up request failed: {e}[/bold red]")
        console.print("--- Warm-up Complete ---")

    # Benchmark Loop
    console.print("\n--- Starting E2E Latency & RTFx Benchmark (Online Pipeline) ---")
    
    results_summary = []

    for audio_len_s, concurrencies in CONCURRENT_BENCHMARK_SCENARIOS.items():
        file_path = SHORT_AUDIO_FILES.get(audio_len_s)
        if not file_path or not os.path.exists(file_path):
            console.print(f"[yellow]Warning: Audio file for {audio_len_s}s not found. Skipping.[/yellow]")
            continue

        # Load audio bytes once
        with open(file_path, "rb") as f:
            audio_bytes = f.read()

        # For RTFx calculation
        single_file_duration = float(audio_len_s) 
        
        for num_concurrent in concurrencies:
            console.print(f"\n[cyan]Scenario: {audio_len_s}s Audio x {num_concurrent} Concurrent Requests[/cyan]")
            
            latencies = []
            
            # Start Benchmark Timer (Total Wall Time)
            start_benchmark = time.perf_counter()
            
            # Submit all requests rapidly
            futures_map = []
            for _ in range(num_concurrent):
                req_start = time.perf_counter()
                future = pipeline(audio_bytes)
                futures_map.append((future, req_start))
            
            # Wait for results and calculate individual E2E latency
            for future, req_start in futures_map:
                try:
                    result = future.result() # Blocking wait
                    req_end = time.perf_counter()
                    latencies.append(req_end - req_start)
                except Exception as e:
                    logger.error(f"Request failed: {e}")

            end_benchmark = time.perf_counter()
            
            # Metrics Calculation
            total_wall_time = end_benchmark - start_benchmark
            total_audio_duration = single_file_duration * num_concurrent
            
            # RTFx
            rtfx = total_audio_duration / total_wall_time if total_wall_time > 0 else 0
            
            # Latency Statistics
            if latencies:
                latencies_np = np.array(latencies)
                min_lat = np.min(latencies_np)
                max_lat = np.max(latencies_np)
                mean_lat = np.mean(latencies_np)
                p50 = np.percentile(latencies_np, 50)
                p90 = np.percentile(latencies_np, 90)
                p99 = np.percentile(latencies_np, 99)
            else:
                min_lat = max_lat = mean_lat = p50 = p90 = p99 = 0.0

            # Print Scenario Result
            print(f"  Total Wall Time: {total_wall_time:.4f}s")
            print(f"  RTFx:            {rtfx:.2f}x")
            print(f"  Latency (s) -> Min: {min_lat:.3f}, Mean: {mean_lat:.3f}, Max: {max_lat:.3f}")
            print(f"                 P50: {p50:.3f}, P90: {p90:.3f}, P99: {p99:.3f}")

            results_summary.append({
                "len": f"{audio_len_s}",
                "conc": num_concurrent,
                "rtfx": rtfx,
                "p50": p50,
                "p90": p90,
                "p99": p99
            })

    # Final Summary Table (Reverted to P50/P90/P99 as requested)
    table = Table(title="E2E Latency & RTFx Summary (Online Pipeline)")
    table.add_column("Audio (s)", justify="center", style="cyan")
    table.add_column("Concurrency", justify="center", style="blue")
    table.add_column("RTFx", justify="right", style="green")
    table.add_column("P50 Latency (s)", justify="right", style="magenta")
    table.add_column("P90 Latency (s)", justify="right", style="magenta")
    table.add_column("P99 Latency (s)", justify="right", style="red")

    for res in results_summary:
        table.add_row(
            str(res["len"]),
            str(res["conc"]),
            f"{res['rtfx']:.2f}x",
            f"{res['p50']:.3f}",
            f"{res['p90']:.3f}",
            f"{res['p99']:.3f}"
        )
    
    console.print("\n")
    console.print(table)
    console.print("\n")

if __name__ == "__main__":
    main()
