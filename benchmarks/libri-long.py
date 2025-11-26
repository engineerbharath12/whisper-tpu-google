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
Benchmark script to reproduce performance numbers for the FlaxWhisperPipline.
"""

import argparse
import os
import time
import numpy as np
import re
import librosa

# ===== Third-Party Library Imports =====
import jax.numpy as jnp
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from transformers.utils import logging

# ===== Local/Project Imports =====
from whisper_jax.pipeline_long_audio import create_pipeline as create_long_audio_pipeline

# --- Configuration ---
console = Console()
logging.set_verbosity_error()

# --- Audio File Paths ---
AUDIO_BASE_PATH = "/home/brathinam_google_com/14Oct/whisper-on-jax/asr_audio_new"
SHORT_AUDIO_FILE = os.path.join(AUDIO_BASE_PATH, "18s", "medical_domain_test.wav")
LONG_AUDIO_FILE = "/home/brathinam_google_com/14Oct/whisper-on-jax/benchmarks/videoplayback.wav"

def run_long_file_benchmark(pipeline, concurrency: int):
    """Tests the pipeline processing for a single long audio file with a given concurrency."""
    SPEED_FACTOR = 3.0
    console.print(Panel(f"[bold blue]Test: Long-File Benchmark (On-the-fly Speed {SPEED_FACTOR}x) with Concurrency: {concurrency}[/bold blue]", expand=False))

    if not os.path.exists(LONG_AUDIO_FILE):
        console.print(f"[bold red]❌ ERROR: Long audio file not found at {LONG_AUDIO_FILE}. Skipping test.[/bold red]")
        return

    total_audio_duration_s = librosa.get_duration(path=LONG_AUDIO_FILE) * concurrency
    console.print(f"\n--- 📊 Starting Benchmark Run ({concurrency} concurrent file(s), {total_audio_duration_s:.2f}s total audio) ---")
    
    benchmark_files = [LONG_AUDIO_FILE] * concurrency
    
    start_time = time.time()
    # Pass speed_factor to the pipeline directly
    results = pipeline(
        benchmark_files, 
        task="transcribe", 
        return_timestamps=False, 
        stride_length_s=2.0,
        speed_factor=SPEED_FACTOR
    )
    total_time = time.time() - start_time

    rtfx = total_audio_duration_s / total_time if total_time > 0 else float("inf")

    table = Table(title=f"FlaxWhisperPipline Performance (Long File - {SPEED_FACTOR}x Speed)")
    table.add_column("Concurrent Files", justify="center", style="blue")
    table.add_column("Audio Length (s)", justify="center", style="cyan")
    table.add_column("Total Time (s)", justify="right", style="magenta")
    table.add_column("RTFx", justify="right", style="green")

    table.add_row(str(concurrency), f"{total_audio_duration_s:.1f}", f"{total_time:.4f}", f"{rtfx:.2f}x")
    console.print(table)
    console.print(f"First 400 chars of transcription: '{results[0].get('text', '')[:400]}...'")

    # --- End of Main Function ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the Whisper JAX pipelines.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="The Hugging Face model ID.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=80,
        help="Batch size to use for inference.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent files to process.",
    )
    args = parser.parse_args()

    # ===== JAX CACHE CONFIGURATION =====
    try:
        from jax import config
        import warnings
        JAX_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".jax_cache")
        os.makedirs(JAX_CACHE_DIR, exist_ok=True)
        config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)
        console.print(f"--- JAX Cache enabled. Using directory: {JAX_CACHE_DIR} ---")
    except ImportError:
        warnings.warn("Could not configure JAX cache.")
    # ===== END OF JAX CACHE CONFIGURATION =====

    try:
        console.print(Panel(f"[bold green]Running Benchmark for Model: {args.model_id}[/bold green]", expand=False))

        console.print(f"--- Instantiating Long Audio Pipeline ---")
        long_audio_pipeline = create_long_audio_pipeline(
            checkpoint=args.model_id,
            dtype=jnp.bfloat16,
            batch_size=args.batch_size,
            encoder_attention_implementation="splash",
            skip_special_tokens=True,
        )

        console.print(f"\n--- Performing Comprehensive Pipeline Warm-up (JIT Compilation) ---")
        
        # Use the 18s audio file for warm-up
        if not os.path.exists(SHORT_AUDIO_FILE):
            console.print("[bold red]Error: 18s audio file for warm-up not found. Skipping warm-up.[/bold red]")
        else:
            for bucket_size in long_audio_pipeline.BATCH_BUCKETS:
                console.print(f"    -> Warming up with a batch size of {bucket_size}...")
                dummy_batch = [SHORT_AUDIO_FILE] * bucket_size
                try:
                    long_audio_pipeline(dummy_batch, task="transcribe", return_timestamps=False)
                except Exception as e:
                    console.print(f"[bold red]    -> Warm-up for batch size {bucket_size} failed: {e}[/bold red]")

        console.print("--- Comprehensive Warm-up Complete ---")

        # Run only the long file benchmark
        run_long_file_benchmark(long_audio_pipeline, args.concurrency)
        
        console.print("\n" * 5)

    except Exception as e:
        console.print(f"[bold red]An error occurred during setup or benchmarking: {e}[/bold red]")
        console.print_exception()