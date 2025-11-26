# ===== Standard Library Imports =====
import argparse
import os
import time
import glob
import subprocess
import tempfile
import shutil

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
AUDIO_BASE_PATH = "/home/brathinam_google_com/14Oct/whisper-on-jax/asr_audio_new"
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

def create_speedup_temp_file(input_path, speed_factor=2.0):
    """
    Creates a temporary audio file with the audio speeded up by the given factor.
    Returns the path to the temporary file.
    """
    if speed_factor == 1.0:
        return input_path

    # Create a temporary file
    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd) # Close the file descriptor, we just need the path

    try:
        # Construct ffmpeg command
        cmd = [
            "ffmpeg",
            "-y", # Overwrite output file if it exists
            "-i", input_path,
            "-filter:a", f"atempo={speed_factor}",
            temp_path
        ]
        
        # Run ffmpeg
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return temp_path
    except subprocess.CalledProcessError:
        console.print(f"[red]Error: Failed to speed up audio file {input_path}[/red]")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return input_path # Fallback to original if failed

def run_concurrent_short_file_benchmark(pipeline):
    """Tests the pipeline with concurrent requests for short audio files."""
    SPEED_FACTOR = 1.0
    console.print(Panel(f"[bold blue]Test: Concurrent Short-File Benchmark (Speed Factor: {SPEED_FACTOR})[/bold blue]", expand=False))
    results_data = []

    # Hardcoded speed factor as requested
    # SPEED_FACTOR = 2.0 (Moved to top of function)

    console.print("\n--- Starting Concurrent Benchmark Runs ---")
    for audio_len_s, concurrencies in CONCURRENT_BENCHMARK_SCENARIOS.items():
        file_path = SHORT_AUDIO_FILES.get(audio_len_s)
        if not file_path or not os.path.exists(file_path):
            console.print(f"[yellow]Warning: Audio file for {audio_len_s}s not found. Skipping.[/yellow]")
            continue

        # Get original duration for RTFx calculation
        audio_duration_sec = librosa.get_duration(path=file_path)
        
        # Create a single sped-up temp file to use for this batch
        # (We avoid creating N temp files to save disk I/O overhead during the benchmark loop itself)
        temp_speedup_file = create_speedup_temp_file(file_path, speed_factor=SPEED_FACTOR)
        is_temp = temp_speedup_file != file_path

        try:
            for num_concurrent_files in concurrencies:
                console.print(f"Testing {audio_len_s}s audio (Original) with concurrent files: [cyan]{num_concurrent_files}[/cyan]...")
                
                # Calculate total audio based on ORIGINAL duration
                total_audio_s = audio_duration_sec * num_concurrent_files
                
                # Use the sped-up file for processing
                benchmark_files = [temp_speedup_file] * num_concurrent_files

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
                    file_name = os.path.basename(file_path) # Show original name
                    transcription = pipeline_results[0].get('text', '[ERROR]')
                    console.print(f"  - [yellow]{file_name} (2x speed)[/yellow]: {transcription}")
                    console.print("-" * 20 + "\n")
        
        finally:
            # clean up temp file
            if is_temp and os.path.exists(temp_speedup_file):
                os.remove(temp_speedup_file)


    table = Table(title=f"FlaxWhisperPipline Performance (Concurrent Short Files - Speed {SPEED_FACTOR}x)")
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
        default="openai/whisper-large-v3-turbo", # Defaulting to Turbo as per recent context
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
