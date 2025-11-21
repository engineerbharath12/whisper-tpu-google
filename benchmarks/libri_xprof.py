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
Benchmark script to trace warm inference for v3 and v3-turbo models.
"""

# ===== Standard Library Imports =====
import os
import time
from datetime import datetime

# ===== Third-Party Library Imports =====
import jax
import jax.numpy as jnp
import librosa
from rich.console import Console
from rich.panel import Panel

# ===== Local/Project Imports =====
from whisper_jax.pipeline_long_audio import create_pipeline as create_long_audio_pipeline
from whisper_jax.pipeline import FlaxWhisperPmapPipeline

# --- Configuration ---
console = Console()
LONG_AUDIO_FILE = "/home/brathinam_google_com/22sept/whisper-on-jax/benchmarks/videoplayback.wav"
SHORT_AUDIO_FILE = "/home/brathinam_google_com/22sept/whisper-on-jax/asr_audio/18s/medical_domain_test.wav"

def run_benchmark(pipeline: FlaxWhisperPmapPipeline, model_id: str, root_trace_dir: str):
    """Runs a benchmark for a single long audio file and captures an xprof trace."""
    scenario_name = model_id.replace("/", "_") + "_warm_inference"
    console.print(Panel(f"[bold blue]Test: Tracing for {scenario_name}[/bold blue]", expand=False))

    if not os.path.exists(LONG_AUDIO_FILE):
        console.print(f"[bold red]❌ ERROR: Audio file not found at {LONG_AUDIO_FILE}. Skipping test.[/bold red]")
        return
    if not os.path.exists(SHORT_AUDIO_FILE):
        console.print(f"[bold red]❌ ERROR: Audio file not found at {SHORT_AUDIO_FILE}. Skipping test.[/bold red]")
        return

    # --- Comprehensive Warm-up ---
    console.print(f"\n--- 🌡️ Performing Comprehensive Pipeline Warm-up (JIT Compilation) ---")
    BUCKET_SIZES = [4, 40, 80]
    for bucket_size in BUCKET_SIZES:
        console.print(f"    -> Warming up with a batch size of {bucket_size}...")
        dummy_batch = [SHORT_AUDIO_FILE] * bucket_size
        try:
            pipeline(dummy_batch, task="transcribe")
        except Exception as e:
            console.print(f"[bold red]    -> Warm-up for batch size {bucket_size} failed: {e}[/bold red]")
    console.print("--- ✅ Comprehensive Warm-up Complete ---")


    # --- Traced Run ---
    trace_path = os.path.join(root_trace_dir, scenario_name)
    console.print(f"\n--- 📊 Starting Traced Benchmark Run ---")
    console.print(f"--- 💾 Saving trace to: {trace_path} ---")
    
    jax.profiler.start_trace(trace_path)
    start_time = time.time()
    
    with jax.profiler.TraceAnnotation("pipeline_execution"):
        results = pipeline([LONG_AUDIO_FILE], task="transcribe")
        
    total_time = time.time() - start_time
    jax.profiler.stop_trace()

    total_audio_duration_s = librosa.get_duration(path=LONG_AUDIO_FILE)
    rtfx = total_audio_duration_s / total_time if total_time > 0 else float("inf")

    console.print(f"\n--- ✅ Tracing Complete ---")
    console.print(f"Total Time: {total_time:.4f}s")
    console.print(f"RTFx: {rtfx:.2f}x")
    console.print(f"Transcription: '{results[0].get('text', '')[:200]}...'")


if __name__ == "__main__":
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
        # --- Create a timestamped directory for the traces ---
        timestamp = datetime.now().strftime("%d%m-%H%M")
        root_trace_dir = f"benchmarks/xprof_traces_collection/{timestamp}"
        os.makedirs(root_trace_dir, exist_ok=True)
        console.print(f"--- 💾 Root trace directory created: {root_trace_dir} ---")

        scenarios = [
            {"model_id": "openai/whisper-large-v3", "feature_extraction": "tpu"},
            {"model_id": "openai/whisper-large-v3-turbo", "feature_extraction": "cpu"},
        ]

        for scenario in scenarios:
            model_id = scenario["model_id"]
            fe_method = scenario["feature_extraction"]
            
            console.print(Panel(f"[bold green]Processing Model: {model_id} with {fe_method.upper()} FE[/bold green]", expand=False))

            console.print(f"--- 🚀 Instantiating Pipeline for {model_id} ---")
            pipeline = create_long_audio_pipeline(
                checkpoint=model_id,
                dtype=jnp.bfloat16,
                batch_size=4, # A small batch size is fine for single file processing
                feature_extraction=fe_method,
                encoder_attention_implementation="splash",
            )

            run_benchmark(pipeline, model_id, root_trace_dir)
            console.print("\n" * 2)

    except Exception as e:
        console.print(f"[bold red]An error occurred: {e}[/bold red]")
        console.print_exception()
