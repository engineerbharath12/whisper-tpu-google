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
Validation script for the FlaxWhisperPipline.
"""

# ===== Standard Library Imports =====
import argparse
import json
import os
import logging

# ===== Third-Party Library Imports =====
import jax
import jax.numpy as jnp
from rich.console import Console
from rich.panel import Panel
from flax import jax_utils
# from transformers.utils import logging as hf_logging

# ===== Local/Project Imports =====
from whisper_jax.pipeline_long_audio import create_pipeline as create_long_audio_pipeline

# --- Configuration ---
console = Console()
# hf_logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main(args):
    console.print(Panel(f"[bold green]Instantiating Pipeline for {args.model_id}[/bold green]", expand=False))

    # Instantiate the long audio pipeline with specified settings
    pipeline = create_long_audio_pipeline(
        checkpoint=args.model_id,
        dtype=jnp.bfloat16,
        batch_size=80,
        encoder_attention_implementation="splash",
        skip_special_tokens=True,
    )

    # --- Verification Step ---
    console.print("\n--- 🔬 Verifying weight data type ---")
    unreplicated_params = jax_utils.unreplicate(pipeline.params)
    first_weight_leaf = jax.tree_util.tree_leaves(unreplicated_params)[0]
    actual_dtype = first_weight_leaf.dtype
    console.print(f"Data type of the first weight tensor is: [bold]{actual_dtype}[/bold]")

    if actual_dtype == jnp.float32:
        console.print("[bold green]✅ SUCCESS: Model weights were correctly upcast to float32.[/bold green]")
    else:
        console.print(f"[bold red]❌ FAILURE: Expected float32, but got {actual_dtype}.[/bold red]")

    audio_path = "/home/brathinam_google_com/22sept/whisper-on-jax/benchmarks/videoplayback.wav"

    if not os.path.exists(audio_path):
        console.print(f"[bold red]❌ ERROR: Audio file not found at {audio_path}.[/bold red]")
        return

    console.print(f"\n--- 🗣️  Running validation for: {audio_path} ---")
    
    transcription = pipeline(audio_path)

    output_path = "/home/brathinam_google_com/22sept/whisper-on-jax/benchmarks/validation_transcription.txt"
    with open(output_path, "w") as f:
        f.write(transcription['text'])
    console.print(f"\n[bold green]✅ Transcription successfully written to {output_path}[/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="openai/whisper-large-v3",
        help="Hugging Face model identifier.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=80,
        help="Batch size to use for inference.",
    )
    args = parser.parse_args()

    # ===== JAX CACHE CONFIGURATION =====
    try:
        from jax import config
        import warnings
        JAX_CACHE_DIR = "/home/brathinam_google_com/22sept/whisper-on-jax/benchmarks/.jax_cache"
        os.makedirs(JAX_CACHE_DIR, exist_ok=True)
        config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)
        console.print(f"--- JAX Cache enabled. Using directory: {JAX_CACHE_DIR} ---")
    except ImportError:
        warnings.warn("Could not configure JAX cache.")
    # ===== END OF JAX CACHE CONFIGURATION =====

    main(args)