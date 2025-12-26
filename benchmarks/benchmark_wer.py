
import argparse
import os
import time
import jiwer
import numpy as np
import sys
import jax.numpy as jnp
from whisper_jax.pipeline_long_audio import create_pipeline as create_long_pipeline
from whisper_jax.pipeline import create_pipeline as create_short_pipeline

def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculates WER with standard normalization using JiWER."""
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
    ])
    
    ref_clean = transformation(reference)
    hyp_clean = transformation(hypothesis)
    
    wer = jiwer.wer(ref_clean, hyp_clean)
    return wer

def run_wer_benchmark(pipeline, audio_files, description="", **pipeline_kwargs):
    """Runs the WER benchmark for a list of audio files."""
    results = {}
    
    print(f"\n=== Running Benchmark: {description} ===")

    for audio_path in audio_files:
        gt_path = audio_path.replace(".wav", ".txt")
        output_path = audio_path.replace(".wav", "_output.txt")
        
        if not os.path.exists(gt_path):
            print(f"  Warning: Ground truth file {gt_path} not found. Skipping.")
            continue
            
        print(f"\n--- Processing: {os.path.basename(audio_path)} ---")
        
        with open(gt_path, "r") as f:
            ground_truth = f.read().strip()
            
        start_time = time.perf_counter()
        
        # --- Run Transcription ---
        try:
            # We pass a list to the pipeline as it expects batches
            # Passing additional kwargs like stride_length_s
            transcription_output = pipeline([audio_path], **pipeline_kwargs)
            hypothesis = transcription_output[0]["text"].strip()
            
            end_time = time.perf_counter()
            e2e_latency = end_time - start_time
            
            # Save transcription to file as requested
            with open(output_path, "w") as f:
                f.write(hypothesis)
            
            # Load back for comparison to strictly follow the "compare text files" instruction
            with open(output_path, "r") as f:
                hypothesis_from_file = f.read().strip()
            
            wer = calculate_wer(ground_truth, hypothesis_from_file)
            
            print(f"  Ground Truth Length: {len(ground_truth)}")
            print(f"  Hypothesis Length:   {len(hypothesis_from_file)}")
            print(f"  E2E Latency:  {e2e_latency:.4f}s")
            print(f"  WER:          {wer * 100:.2f}%")
            results[audio_path] = {"latency": e2e_latency, "wer": wer}
            
        except Exception as e:
            print(f"  Error transcribing {os.path.basename(audio_path)}: {e}")
            results[audio_path] = {"latency": -1, "wer": 1.0} # Indicate error
            
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WER Benchmark for Whisper models.")
    parser.add_argument("--model_id", default="openai/whisper-large-v3-turbo", help="Whisper model ID.")
    args = parser.parse_args()
    
    SHORT_AUDIO = "/home/brathinam_google_com/whisper/WER/audio_files/short_audio.wav"
    LONG_AUDIO = "/home/brathinam_google_com/whisper/WER/audio_files/long_audio.wav"
    
    # Check if files exist
    if not os.path.exists(SHORT_AUDIO) or not os.path.exists(LONG_AUDIO):
        print(f"Error: Audio files not found.")
        # Continue if one exists? No, user wants both.
        # But for robustness let's check individually in the loop call
        pass

    # ===== JAX CACHE CONFIGURATION =====
    try:
        from jax import config
        # Use specific cache directory
        JAX_CACHE_DIR = "/home/brathinam_google_com/whisper/26dec"
        os.makedirs(JAX_CACHE_DIR, exist_ok=True)
        config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)
        config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        config.update("jax_persistent_cache_min_compile_time_secs", 0)
        print(f"--- JAX Cache enabled. Using directory: {JAX_CACHE_DIR} ---")
    except ImportError:
        print("Could not configure JAX cache.")

    # --- Initialize Short Pipeline ---
    # print(f"\n--- Instantiating Short Pipeline (from pipeline.py) for {args.model_id} ---")
    # short_pipeline = create_short_pipeline(
    #     checkpoint=args.model_id, 
    #     dtype=jnp.bfloat16, 
    #     batch_size=80,
    #     encoder_attention_implementation="splash",
    #     skip_special_tokens=True,
    # )
    
    # --- Initialize Long Pipeline ---
    print(f"\n--- Instantiating Long Pipeline (from pipeline_long_audio.py) for {args.model_id} ---")
    long_pipeline = create_long_pipeline(
        checkpoint=args.model_id, 
        dtype=jnp.bfloat16, 
        batch_size=80,
        encoder_attention_implementation="splash",
        skip_special_tokens=True,
    )

    # Warmup (Optional but recommended, using one file for both to be safe)
    print("\n--- Performing Warm-up ---")
    warmup_file = "/home/brathinam_google_com/whisper/26dec/whisper-tpu-google/asr_audio_16K/18s/medical_domain_test.wav"
    # if not os.path.exists(warmup_file):
    #     warmup_file = os.path.join(os.getcwd(), warmup_file)
        
    if os.path.exists(warmup_file):
        # print(f"    -> Warming up Short Pipeline...")
        # short_pipeline([warmup_file])
        print(f"    -> Warming up Long Pipeline...")
        long_pipeline([warmup_file])
    else:
        print(f"    -> Warning: Warmup file {warmup_file} not found. Skipping warmup.")
                
    print("--- Warm-up Complete ---")

    # --- Run Benchmark ---
    
    # 1. Short Audio with Short Pipeline
    # if os.path.exists(SHORT_AUDIO):
    #     run_wer_benchmark(short_pipeline, [SHORT_AUDIO], description="Short Audio (pipeline.py)")
    # else:
    #     print(f"Short audio file not found: {SHORT_AUDIO}")

    # 2. Long Audio with Long Pipeline
    if os.path.exists(LONG_AUDIO):
        run_wer_benchmark(
            long_pipeline, 
            [LONG_AUDIO], 
            description="Long Audio (pipeline_long_audio.py)"
        )
    else:
        print(f"Long audio file not found: {LONG_AUDIO}")

