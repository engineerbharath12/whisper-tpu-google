# benchmarks/test_preprocessing_speed.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import librosa
from rich.console import Console
from rich.panel import Panel

# Correctly import the pipeline
from whisper_jax import pipeline_long_audio_instrumented

# --- Configuration ---
console = Console()
model_id = "openai/whisper-large-v3"
LONG_AUDIO_FILE = "/home/brathinam_google_com/22sept/whisper-on-jax/benchmarks/videoplayback.wav"
# Create a short, 20-second version of the long file for a quick test
SHORT_TEST_FILE = "/tmp/videoplayback_20s.wav"

def run_test(pipeline, concurrency: int):
    console.print(Panel(f"[bold blue]Testing Preprocessing with Concurrency: {concurrency}[/bold blue]", expand=False))
    
    # Use a list of the same short file for the test
    benchmark_files = [SHORT_TEST_FILE] * concurrency
    
    # We only care about the logs, but we need to run the pipeline
    pipeline(benchmark_files, task="transcribe", return_timestamps=False)


if __name__ == "__main__":
    # Create the 20-second test file
    if not os.path.exists(SHORT_TEST_FILE):
        console.print(f"Creating short test file: {SHORT_TEST_FILE}")
        y, sr = librosa.load(LONG_AUDIO_FILE, sr=16000, duration=20.0)
        import soundfile as sf
        sf.write(SHORT_TEST_FILE, y, sr)

    try:
        pipeline = pipeline_long_audio_instrumented.create_pipeline(checkpoint=model_id)

        # --- Run for Single Concurrency ---
        run_test(pipeline, 1)
        
        console.print("\n" * 2)
        
        # --- Run for 10 Concurrency ---
        run_test(pipeline, 10)

    except Exception as e:
        console.print(f"[bold red]An error occurred during the test: {e}[/bold red]")
        console.print_exception()
    finally:
        # Clean up the temporary instrumented file
        os.remove("/home/brathinam_google_com/14Oct/whisper-on-jax/pipeline_long_audio_instrumented.py")