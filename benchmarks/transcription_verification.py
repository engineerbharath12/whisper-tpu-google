import time
import warnings
import jax
import jax.numpy as jnp
import librosa
from rich.console import Console

from whisper_jax.pipeline import create_pipeline

# --- Suppress informational warnings from the transformers library ---
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# --- Benchmark Configuration ---
MODEL_ID = "openai/whisper-large-v3"
DTYPE = jnp.bfloat16
BATCH_SIZE = 32 # Internal batch size for chunking within the pipeline
SAMPLING_RATE = 16000
MAX_LENGTH = 448

# Define the audio files to test
AUDIO_FILES = {
    2: "./asr_audio/2s/good_day.wav",
    8: "./asr_audio/8s/462_gu.wav",
    14: "./asr_audio/14s/hindi 15 sec.wav",
    18: "./asr_audio/18s/medical_domain_test.wav",
}

def run_transcription_test(pipeline, audio_files_list, test_name="Batch Transcription Test"):
    """Runs a transcription test for a list of audio files and prints the results."""
    console = Console()
    console.print(f"\n--- Running {test_name} for {len(audio_files_list)} audio files ---")

    total_audio_duration = sum(librosa.get_duration(path=f) for f in audio_files_list)

    start_time = time.time()
    outputs = pipeline(audio_files_list, task="transcribe", max_length=MAX_LENGTH)
    jax.device_put(0.0).block_until_ready()
    total_time = time.time() - start_time

    rtfx = total_audio_duration / total_time if total_time > 0 else 0.0

    transcriptions = [output['text'] for output in outputs] if outputs else []
    
    for i, text in enumerate(transcriptions):
        console.print(f"    File {i+1} Transcription: \"{text.strip()}\"")
    console.print(f"    Total Time taken for batch: {total_time:.2f}s, RTFx: {rtfx:.2f}x")
    
    return transcriptions

def main():
    console = Console()
    console.print(f"--- Starting Transcription Verification for: {MODEL_ID} ---")

    console.print("\nInitializing transcription pipeline...")
    pmap_pipeline = create_pipeline("pmap", checkpoint=MODEL_ID, dtype=DTYPE, batch_size=BATCH_SIZE)

    # Warm-up inference
    console.print("\n--- Performing warm-up inference (18s audio) ---")
    warm_up_audio_path = AUDIO_FILES[18]
    run_transcription_test(pmap_pipeline, [warm_up_audio_path], "Warm-up")
    
    console.print("\n--- Running individual transcription tests for various audio lengths ---")
    for audio_len_s, audio_file_path in AUDIO_FILES.items():
        run_transcription_test(pmap_pipeline, [audio_file_path], f"Individual {audio_len_s}s Audio Test")

    console.print("\n--- Running batch transcription test ---")
    all_audio_files = list(AUDIO_FILES.values())
    run_transcription_test(pmap_pipeline, all_audio_files, "Full Batch Transcription Test")

    # Add test for 200 concurrent 18s audio files
    console.print("\n--- Running 200 Concurrent 18s Audio Test ---")
    audio_18s_path = AUDIO_FILES[18]
    concurrent_18s_audio_files = [audio_18s_path] * 200
    run_transcription_test(pmap_pipeline, concurrent_18s_audio_files, "200 Concurrent 18s Audio Test")

    # Add test for 200 concurrent 2s audio files
    console.print("\n--- Running 200 Concurrent 2s Audio Test ---")
    audio_2s_path = AUDIO_FILES[2]
    concurrent_2s_audio_files = [audio_2s_path] * 200
    run_transcription_test(pmap_pipeline, concurrent_2s_audio_files, "200 Concurrent 2s Audio Test")
    
    console.print("\n--- Transcription verification complete. ---")

    del pmap_pipeline
    jax.clear_caches()

if __name__ == "__main__":
    main()
