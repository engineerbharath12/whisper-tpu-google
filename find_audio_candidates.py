import os
import subprocess
import glob

SEARCH_DIR = "/home/brathinam_google_com/whisper/WER/LibriSpeech"
TARGET_SHORT_MIN = 17.5
TARGET_SHORT_MAX = 18.5
TARGET_LONG_MIN = 300.0

def get_duration(file_path):
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    except Exception:
        return 0.0

def find_transcript(audio_path):
    dirname = os.path.dirname(audio_path)
    filename = os.path.basename(audio_path)
    file_id = filename.replace(".flac", "")
    
    # Standard LibriSpeech: Look for *.trans.txt in the same dir
    trans_files = glob.glob(os.path.join(dirname, "*.trans.txt"))
    for tf in trans_files:
        with open(tf, 'r') as f:
            for line in f:
                if line.startswith(file_id):
                    # Format: ID TRANSCRIPT...
                    return line.split(" ", 1)[1].strip()
    return None

def search():
    found_short = False
    found_long = False
    
    print(f"Searching in {SEARCH_DIR}...")
    
    # We prioritize checking by file size to quickly find candidates
    # Short ~18s -> approx 300KB - 600KB (varies)
    # Long > 5m -> approx > 4MB
    
    candidate_short = []
    candidate_long = []

    count = 0
    for root, dirs, files in os.walk(SEARCH_DIR):
        for file in files:
            if not file.endswith(".flac"):
                continue
                
            path = os.path.join(root, file)
            size = os.path.getsize(path)
            
            # Optimization: Only probe likely candidates
            
            # Check for Long Candidate (> 4MB)
            if not found_long and size > 4 * 1024 * 1024:
                dur = get_duration(path)
                if dur >= TARGET_LONG_MIN:
                    trans = find_transcript(path)
                    if trans:
                        print(f"\n[FOUND LONG] {path}")
                        print(f"  Duration: {dur}s")
                        print(f"  Transcript: {trans[:100]}...")
                        found_long = True
            
            # Check for Short Candidate (18s is roughly 300-600KB)
            if not found_short and 250 * 1024 < size < 700 * 1024:
                dur = get_duration(path)
                if TARGET_SHORT_MIN <= dur <= TARGET_SHORT_MAX:
                    trans = find_transcript(path)
                    if trans:
                        print(f"\n[FOUND SHORT] {path}")
                        print(f"  Duration: {dur}s")
                        print(f"  Transcript: {trans[:100]}...")
                        found_short = True
            
            count += 1
            if count % 500 == 0:
                print(f"Checked {count} files...")
                
            if found_short and found_long:
                return

    if not found_short:
        print("\nCould not find a perfect 18s file in the scanned files.")
    if not found_long:
        print("\nCould not find a >5min file in the scanned files (LibriSpeech is typically segmented).")

if __name__ == "__main__":
    search()
