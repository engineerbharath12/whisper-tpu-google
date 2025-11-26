import requests
import time
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.table import Table
import librosa
import subprocess

# --- Configuration ---
console = Console()
SERVER_URL = "http://127.0.0.1:8000"
READY_ENDPOINT = f"{SERVER_URL}/ready"
TRANSCRIBE_ENDPOINT = f"{SERVER_URL}/transcribe"
# Define the set of audio files to randomly pick from for each request
AUDIO_FILE = "/home/brathinam_google_com/14Oct/whisper-on-jax/asr_audio_new/18s/medical_domain_test.wav"

def wait_for_server_ready():
    """Polls the /ready endpoint until the server is fully initialized."""
    console.print("[bold yellow]Waiting for server to become ready...[/bold yellow]")
    while True:
        try:
            response = requests.get(READY_ENDPOINT, timeout=5)
            if response.status_code == 200:
                console.print("[bold green]✅ Server is ready![/bold green]")
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(5)

def apply_speed_to_bytes(audio_bytes, speed):
    if speed == 1.0:
        return audio_bytes
    try:
        # Process in-memory bytes using ffmpeg pipe
        cmd = [
            "ffmpeg", "-i", "pipe:0", 
            "-filter:a", f"atempo={speed}", 
            "-f", "wav", "pipe:1"
        ]
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out, _ = process.communicate(input=audio_bytes)
        return out
    except Exception as e:
        console.print(f"[red]Speed up failed: {e}[/red]")
        return audio_bytes

def transcribe_worker(file_path, speed):
    """Reads file from disk, processes it, and uploads."""
    file_name = os.path.basename(file_path)
    
    # 1. Simulate Real Client: Read from Disk
    with open(file_path, "rb") as f:
        raw_bytes = f.read()
        
    # 2. Client-side Processing (Speed Up)
    final_bytes = apply_speed_to_bytes(raw_bytes, speed)
    
    # 3. Upload
    files = {"file": (file_name, final_bytes, "audio/wav")}
    
    # Timeout increased to ensure we capture long-tail latencies under load
    response = requests.post(TRANSCRIBE_ENDPOINT, files=files, params={"task": "transcribe"}, timeout=1200)
    response.raise_for_status()
    return response.json()

def run_api_benchmark(concurrency: int, speed: float):
    """Sends multiple audio files to the API concurrently and prints performance."""
    console.print(f"\n[bold blue]Submitting {concurrency} concurrent transcription request(s) (Speed: {speed}x)...[/bold blue]")
    console.print(f"File: {os.path.basename(AUDIO_FILE)}")
    console.print(f"[dim]Mode: Individual Disk Read -> FFmpeg -> Upload[/dim]")

    total_start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for _ in range(concurrency):
            # Each worker reads the file independently
            futures.append(executor.submit(transcribe_worker, AUDIO_FILE, speed))
        
        results = []
        failed_count = 0
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                failed_count += 1
                if failed_count <= 5:
                    console.print(f"[bold red]❌ A request failed: {e}[/bold red]")

    total_end_time = time.time()
    total_wall_time = total_end_time - total_start_time
    
    if not results:
        console.print("[bold red]No successful transcriptions to report.[/bold red]")
        return

    # --- Display Results ---
    audio_duration_s = librosa.get_duration(path=AUDIO_FILE)
    total_audio_processed_s = audio_duration_s * len(results)
    
    # System RTFx = Total Audio Duration / Wall Time
    system_rtfx = total_audio_processed_s / total_wall_time

    table = Table(title=f"FastAPI Concurrent Transcription Performance (Concurrency: {concurrency})")
    table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    table.add_column("Value", justify="left", style="magenta")

    table.add_row("Successful Requests", str(len(results)))
    table.add_row("Failed Requests", str(failed_count))
    table.add_row("Total Audio Processed (s)", f"{total_audio_processed_s:.2f}")
    table.add_row("Total Wall Time (s)", f"{total_wall_time:.2f}")
    table.add_row("[bold green]System RTFx[/bold green]", f"[bold green]{system_rtfx:.2f}x[/bold green]")
    
    console.print(table)

    transcription = results[0].get("transcription", {}).get("text", "")
    console.print(f"\n[bold]First 400 chars of transcription:[/bold] '{transcription[:400]}...'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the Whisper JAX FastAPI server.")
    parser.add_argument("-c", "--concurrency", type=int, default=1280, help="Number of concurrent requests to send.")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed factor.")
    args = parser.parse_args()

    wait_for_server_ready()
    run_api_benchmark(args.concurrency, args.speed)
