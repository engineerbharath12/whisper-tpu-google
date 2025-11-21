# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Benchmark script to compare the performance of different long-audio pipeline implementations.
"""

import os
import time
import librosa
import jax.numpy as jnp
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from transformers.utils import logging

# --- Import the two pipeline versions ---
from whisper_jax.pipeline_long_audio_seek import create_pipeline as create_seek_pipeline
from whisper_jax.pipeline_long_audio import create_pipeline as create_stream_pipeline

# --- Configuration ---
console = Console()
logging.set_verbosity_error()
model_id = "openai/whisper-large-v3"

# --- Audio File Paths ---
AUDIO_BASE_PATH = "/home/brathinam_google_com/22sept/whisper-on-jax/asr_audio_new"
SHORT_AUDIO_FILE = os.path.join(AUDIO_BASE_PATH, "18s", "medical_domain_test.wav")
LONG_AUDIO_FILE = "/home/brathinam_google_com/22sept/whisper-on-jax/benchmarks/videoplayback.wav"

def run_benchmark(pipeline_creator, pipeline_name: str, concurrency: int):
    """Initializes, warms up, and benchmarks a given pipeline implementation."""
    console.print(Panel(f"[bold blue]Benchmarking Implementation: '{pipeline_name}'[/bold blue]", expand=False))

    # --- Initialization ---
    console.print(f"--- 🚀 Instantiating {pipeline_name} Pipeline ---")
    pipeline = pipeline_creator(checkpoint=model_id)

    # --- Warm-up ---
    console.print(f"\n--- 🌡️ Performing Warm-up ---")
    if not os.path.exists(SHORT_AUDIO_FILE):
        console.print("[bold red]Error: Short audio file for warm-up not found. Skipping warm-up.[/bold red]")
    else:
        for bucket_size in pipeline.BATCH_BUCKETS:
            console.print(f"    -> Warming up with batch size {bucket_size}...")
            dummy_batch = [SHORT_AUDIO_FILE] * bucket_size
            try:
                pipeline(dummy_batch, task="transcribe", return_timestamps=False)
            except Exception as e:
                console.print(f"[bold red]    -> Warm-up for batch size {bucket_size} failed: {e}[/bold red]")
    console.print("--- ✅ Warm-up Complete ---")

    # --- Benchmark Execution ---
    if not os.path.exists(LONG_AUDIO_FILE):
        console.print(f"[bold red]❌ ERROR: Long audio file not found at {LONG_AUDIO_FILE}. Skipping test.[/bold red]")
        return 0, 0, "File not found"

    total_audio_duration_s = librosa.get_duration(path=LONG_AUDIO_FILE) * concurrency
    console.print(f"\n--- 📊 Starting Benchmark ({concurrency} concurrent files, {total_audio_duration_s:.2f}s total audio) ---")
    
    benchmark_files = [LONG_AUDIO_FILE] * concurrency
    
    start_time = time.time()
    results = pipeline(benchmark_files, task="transcribe", return_timestamps=False)
    total_time = time.time() - start_time

    rtfx = total_audio_duration_s / total_time if total_time > 0 else float("inf")
    
    console.print(f"First 400 chars of transcription: '{results[0].get('text', '')[:400]}'...")
    console.print(f"[bold green]Benchmark Complete. Total Time: {total_time:.4f}s, RTFx: {rtfx:.2f}x[/bold green]\n")
    
    return total_time, rtfx, results[0].get('text', '')


if __name__ == "__main__":
    # ===== JAX CACHE CONFIGURATION =====
    try:
        from jax import config
        JAX_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".jax_cache")
        os.makedirs(JAX_CACHE_DIR, exist_ok=True)
        config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)
        console.print(f"--- JAX Cache enabled. Using directory: {JAX_CACHE_DIR} ---")
    except ImportError:
        console.print("[yellow]Warning: Could not configure JAX cache.[/yellow]")
    # ===== END OF JAX CACHE CONFIGURATION =====

    try:
        console.print(Panel(f"[bold green]Running Benchmark Comparison for Model: {model_id}[/bold green]", expand=False))

        pipelines_to_test = [
            ("Seek-Based (Original)", create_seek_pipeline),
            ("Streaming (New)", create_stream_pipeline),
        ]

        results_data = []
        for name, creator in pipelines_to_test:
            total_time, rtfx, transcription = run_benchmark(creator, name, 10)
            results_data.append([name, f"{total_time:.4f}", f"{rtfx:.2f}x"])

        # --- Final Results Table ---
        table = Table(title="Benchmark Comparison: Long Audio Pipelines")
        table.add_column("Implementation", justify="left", style="cyan")
        table.add_column("Total Time (s)", justify="right", style="magenta")
        table.add_column("RTFx", justify="right", style="green")

        for row in results_data:
            table.add_row(*row)
        
        console.print(table)

    except Exception as e:
        console.print(f"[bold red]An error occurred during the benchmark: {e}[/bold red]")
        console.print_exception()
