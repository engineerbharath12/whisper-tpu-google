# ===== Standard Library Imports =====
import argparse
import os
import time
import glob

# ===== Third-Party Library Imports =====
import jax.numpy as jnp
import librosa
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from transformers.utils import logging

# ===== Local/Project Imports =====
from whisper_jax.pipeline import create_pipeline

# --- Configuration ---
console = Console()
logging.set_verbosity_error()
AUDIO_BASE_PATH = "/home/brathinam_google_com/14Oct/whisper-on-jax/asr_audio"
WARMUP_FILE = os.path.join(AUDIO_BASE_PATH, "18s", "medical_domain_test.wav")

# --- Test Scenarios ---
CONCURRENT_BENCHMARK_SCENARIOS = {
    2: [1, 50, 200],
    8: [1, 50, 200],
    14: [1, 50, 200],
    18: [1, 50, 200, 400, 800, 1280],
}

SHORT_AUDIO_FILES = {
    2: os.path.join(AUDIO_BASE_PATH, "2s", "good_day.wav"),
    8: os.path.join(AUDIO_BASE_PATH, "8s", "462_gu.wav"),
    14: os.path.join(AUDIO_BASE_PATH, "14s", "hindi_15_sec.wav"),
    18: os.path.join(AUDIO_BASE_PATH, "18s", "medical_domain_test.wav"),
}

def run_concurrent_short_file_benchmark(pipeline):
    """Tests the pipeline with concurrent requests for short audio files."""
    console.print(Panel("[bold blue]Test: Concurrent Short-File Benchmark[/bold blue]", expand=False))
    results_data = []

    console.print("\n--- 📊 Starting Concurrent Benchmark Runs ---")
    for audio_len_s, concurrencies in CONCURRENT_BENCHMARK_SCENARIOS.items():
        file_path = SHORT_AUDIO_FILES.get(audio_len_s)
        if not file_path or not os.path.exists(file_path):
            console.print(f"[yellow]Warning: Audio file for {audio_len_s}s not found. Skipping.[/yellow]")
            continue

        audio_duration_sec = librosa.get_duration(path=file_path)

        for num_concurrent_files in concurrencies:
            console.print(f"Testing {audio_len_s}s audio with concurrent files: [cyan]{num_concurrent_files}[/cyan]...")
            total_audio_s = audio_duration_sec * num_concurrent_files
            benchmark_files = [file_path] * num_concurrent_files

            start_time = time.time()
            pipeline_results = pipeline(benchmark_files, task="transcribe")
            total_processing_time = time.time() - start_time

            rtfx = total_audio_s / total_processing_time if total_processing_time > 0 else float("inf")
            
            results_data.append({
                "len": f"{audio_len_s:.1f}", 
                "batch": num_concurrent_files, 
                "time": total_processing_time, 
                "rtfx": rtfx
            })
            
            # Print sample transcription for the last iteration of each audio size
            if num_concurrent_files == concurrencies[-1]:
                console.print("\n--- Sample Transcription (Last Iteration) ---")
                file_name = os.path.basename(benchmark_files[0])
                transcription = pipeline_results[0].get('text', '[ERROR]')
                console.print(f"  - [yellow]{file_name}[/yellow]: {transcription}")
                console.print("-" * 20 + "\n")


    table = Table(title="FlaxWhisperPipline Performance (Concurrent Short Files)")
    table.add_column("Audio Length (s)", justify="center", style="cyan")
    table.add_column("Concurrent Files", justify="center", style="blue")
    table.add_column("Total Time (s)", justify="right", style="magenta")
    table.add_column("RTFx", justify="right", style="green")

    for res in results_data:
        table.add_row(res["len"], str(res["batch"]), f"{res['time']:.4f}", f"{res['rtfx']:.2f}x")
    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the Whisper JAX short audio pipeline.")
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="The Hugging Face model ID.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=80,
        help="The batch size to use for the pipeline.",
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

        console.print(f"--- Instantiating Short Audio Pipeline ---")
        pipeline = create_pipeline(
            checkpoint=args.model_id, 
            dtype=jnp.bfloat16, 
            batch_size=args.batch_size,
            encoder_attention_implementation="splash",
            skip_special_tokens=True,
        )
        # --- Enable Early Stopping ---
        # pipeline.model.generation_config.early_stopping = True
        console.print("--- Early Stopping Disabled ---")

        console.print(f"\n--- Performing Comprehensive Pipeline Warm-up (JIT Compilation) ---")
        
        if not os.path.exists(WARMUP_FILE):
            console.print("[bold red]Error: Warm-up file not found. Skipping warm-up.[/bold red]")
        else:
            for bucket_size in pipeline.BATCH_BUCKETS:
                console.print(f"    -> Warming up with a batch size of {bucket_size}...")
                dummy_batch = [WARMUP_FILE] * bucket_size
                try:
                    pipeline(dummy_batch, task="transcribe")
                except Exception as e:
                    console.print(f"[bold red]    -> Warm-up for batch size {bucket_size} failed: {e}[/bold red]")

        console.print("--- Comprehensive Warm-up Complete ---")

        run_concurrent_short_file_benchmark(pipeline)
        
        console.print("\n" * 2)

    except Exception as e:
        console.print(f"[bold red]An error occurred during setup or benchmarking: {e}[/bold red]")
        console.print_exception()