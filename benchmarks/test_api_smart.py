import requests
import time
import os
import argparse
import threading
import queue
import random
import librosa 
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

# --- Configuration ---
console = Console()
SERVER_URL = "http://127.0.0.1:8000"
READY_ENDPOINT = f"{SERVER_URL}/ready"
TRANSCRIBE_ENDPOINT = f"{SERVER_URL}/transcribe"
AUDIO_BASE_DIR = "/home/brathinam_google_com/14Oct/whisper-on-jax/asr_audio_new"

# Audio file to test (Hardcoded to 18s for this test)
AUDIO_TEST_SET = [
    os.path.join(AUDIO_BASE_DIR, "18s", "medical_domain_test.wav"),
]

AUDIO_DATA_MAP = {}

def apply_speed_to_bytes(audio_bytes, speed):
    if speed == 1.0:
        return audio_bytes
    try:
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

def wait_for_server_ready():
    console.print("[bold yellow]Waiting for server to become ready...[/bold yellow]")
    while True:
        try:
            response = requests.get(READY_ENDPOINT, timeout=5)
            if response.status_code == 200:
                console.print("[green]Server is ready![/green]")
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(5)

def load_audio_files():
    console.print(f"[bold cyan]Loading audio files...[/bold cyan]")
    for file_path in AUDIO_TEST_SET:
        if not os.path.exists(file_path):
            console.print(f"[bold red]Error: {file_path} not found.[/bold red]")
            sys.exit(1)
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        duration_s = librosa.get_duration(path=file_path)
        AUDIO_DATA_MAP[file_path] = (audio_bytes, duration_s)
    console.print("[bold cyan]✅ Audio loaded.[/bold cyan]\n")

def transcribe_worker(stats_ref, request_idx, speed=1.0):
    # Pick random file (only 1 in list currently)
    chosen_file_path = random.choice(list(AUDIO_DATA_MAP.keys()))
    audio_bytes, original_duration_s = AUDIO_DATA_MAP[chosen_file_path]
    file_name = os.path.basename(chosen_file_path)
    
    # Client-side Processing
    processed_bytes = apply_speed_to_bytes(audio_bytes, speed)

    try:
        files = {"file": (file_name, processed_bytes, "audio/wav")}
        # No timeout on request to ensure we capture stragglers in heavy load
        # But server might timeout. 
        response = requests.post(
            TRANSCRIBE_ENDPOINT, 
            files=files, 
            params={"task": "transcribe"}, 
            timeout=3600 # 1 hour timeout
        )
        response.raise_for_status()
        result = response.json()
            
        with stats_ref["lock"]:
            stats_ref["successful"] += 1
            if stats_ref["sample_transcription"] is None and "transcription" in result:
                stats_ref["sample_transcription"] = result["transcription"].get("text", "")
            
            # Accumulate ORIGINAL duration
            stats_ref["total_audio_processed"] += original_duration_s
            
            # Pipeline time
            if 'performance' in result:
                stats_ref["total_pipeline_time"] += result['performance'].get('pipeline_execution_time_s', 0)
        
        return result
            
    except Exception as e:
        with stats_ref["lock"]:
            stats_ref["failed"] += 1
        if stats_ref["failed"] <= 5: # Only print first 5 errors
            console.print(f"[red]Request failed: {e}[/red]")
        return None

def run_simultaneous_benchmark(total_requests, speed):
    current_stats = {
        "successful": 0,
        "failed": 0,
        "total_pipeline_time": 0.0,
        "total_audio_processed": 0.0,
        "sample_transcription": None,
        "lock": threading.Lock()
    }

    console.print(f"\n[bold blue]🚀 Launching {total_requests} Simultaneous Requests (Speed: {speed}x)[/bold blue]")
    
    # We use max_workers = total_requests to ensure TRUE simultaneous dispatch
    # This creates 1280 threads. Python can handle this IO-bound workload.
    
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=total_requests) as executor:
        futures = []
        # Dispatch all
        for i in range(total_requests):
            futures.append(executor.submit(transcribe_worker, current_stats, i, speed))
        
        # Wait for all to complete
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
        ) as progress:
            task_id = progress.add_task("[cyan]Processing...", total=total_requests)
            
            for _ in as_completed(futures):
                progress.advance(task_id)

    end_time = time.time()
    wall_time = end_time - start_time
    
    print_results(wall_time, total_requests, current_stats)

def print_results(wall_time, total_requests, stats_data):
    if stats_data["successful"] == 0:
        console.print("[bold red]No successful requests.[/bold red]")
        return

    # System RTFx = Total Audio / Wall Time
    # This is the "Throughput" of the server under load
    system_rtfx = stats_data["total_audio_processed"] / wall_time
    
    table = Table(title=f"Final Benchmark Results")
    table.add_column("Metric", justify="right", style="cyan")
    table.add_column("Value", justify="left", style="magenta")

    table.add_row("Total Requests", str(total_requests))
    table.add_row("Successful", str(stats_data["successful"]))
    table.add_row("Failed", f"[red]{stats_data['failed']}[/red]")
    table.add_row("Total Audio Processed", f"{stats_data['total_audio_processed'] / 60:.1f} min")
    table.add_row("Total Wall Time", f"{wall_time:.2f} s")
    table.add_row("System Throughput (Real RTFx)", f"[bold green]{system_rtfx:.2f}x[/bold green]")
    
    console.print(table)

    if stats_data["sample_transcription"]:
        console.print(f"\n[bold]Sample Transcription:[/bold]\n'{stats_data['sample_transcription'][:200]}...'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--total", type=int, default=1280, help="Total requests (simultaneous)")
    parser.add_argument("--speed", type=float, default=1.0, help="Audio speed factor")
    args = parser.parse_args()

    wait_for_server_ready()
    load_audio_files()
    
    run_simultaneous_benchmark(args.total, args.speed)