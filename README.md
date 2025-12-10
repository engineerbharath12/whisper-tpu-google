# Custom Implementation for Whisper-V3-large for TPUv6e 

This implementation is based on the original [sanchit-gandhi/whisper-jax](https://github.com/sanchit-gandhi/whisper-jax) repository, optimized for **TPU-v6e** with the latest **JAX packages** and the **whisper-v3-large** model family.
For more details on the implementation also read https://medium.com/@engineerbharath/unlocking-supersonic-speech-recognition-a-deep-dive-into-optimizing-whisper-on-tpus-6b8d734993d9
## Supported Models
We have extensively tested and benchmarked the following models:
*   **`openai/whisper-large-v3`**: The standard SOTA model.
*   **`openai/whisper-large-v3-turbo`**: A highly optimized version (4 decoder layers) that delivers ~2x throughput.
    *   *Note:* Native Flax weights are not available for the Turbo model. This repository automatically handles the on-the-fly conversion of PyTorch weights to Flax during initialization.

## Optimized Pipelines
This repository provides three distinct pipeline architectures, each tailored for a specific use case:

### 1. Standard Batch Pipeline (`pipeline.py`)
*   **Use Case:** Offline benchmarking of short audio clips (<30s).
*   **Logic:** High-throughput batch processing using `jax.pmap`. Ideal for maximizing TPU saturation with pre-loaded lists of files.
*   **Usage:** See `benchmarks/libri-short.py`.

### 2. Long Audio Pipeline (`pipeline_long_audio.py`)
*   **Use Case:** Offline transcription of long files (e.g., 1 hour+).
*   **Logic:** Implements chunking with configurable overlap (stride) and timestamp alignment. It ensures context is preserved across 30s boundaries for high accuracy.
*   **Usage:** See `benchmarks/libri-long.py`.

### 3. Online Serving Pipeline (`pipeline_online.py`)
*   **Use Case:** Production API serving (FastAPI).
*   **Logic:** A sophisticated asynchronous engine featuring:
    *   **ProcessPoolExecutor:** Offloads CPU-heavy FFmpeg decoding to bypass the GIL.
    *   **Patient Batcher:** Dynamically groups incoming requests into optimal batches (4, 40, 80) with a configurable timeout.
    *   **TPU Offload:** Feature extraction (Spectrogram) runs on the TPU.
    *   **Low Latency:** Designed for real-time streaming applications.

---

## **1. Infrastructure Setup (Local Machine)**

These steps provision the necessary Google Cloud infrastructure. Run these commands from your local terminal.

### **1.1 Set Project Details** Replace with your environemnt  details 

```
export TPU_NAME=TPU-testing
export PROJECT_ID=tpu-launchpad-playground
export REGION=us-central1
export ZONE=us-central1-b

```

### **1.2 Create TPU VM**

This command provisions a `2x4` TPU v6e slice.

```
gcloud alpha compute tpus tpu-vm create $TPU_NAME \
  --type v6e --topology 2x4 \
  --project $PROJECT_ID --zone $ZONE \
  --version v2-alpha-tpuv6e --tags=allow-iap-ssh

```

### **1.3 Configure Firewall for SSH \-optional for TPU-Playground**

This rule is specific to the playground project to allow IAP access.

```
gcloud compute firewall-rules create allow-iap-on-tagged-vms-high-priority \
  --project=tpu-launchpad-playground \
  --description="High-priority allow for IAP on tagged VMs" \
  --direction=INGRESS \
  --priority=900 \
  --network=default \
  --action=ALLOW \
  --rules=tcp:22 \
  --source-ranges=35.235.240.0/20 \
  --target-tags=allow-iap-ssh

```

### **1.4 SSH into the Machine**

Connect to the TPU VM. All subsequent commands will be run inside this machine.

```
gcloud alpha compute tpus tpu-vm ssh --zone $ZONE $TPU_NAME --project $PROJECT_ID --tunnel-through-iap

```

## **2. Software Installation & Configuration (Inside TPU VM)**

Execute the following steps inside your TPU VM\'s terminal.

### **2.1 Install System Tools**

```
# Install tpu-info & verify setup
pip install tpu-info
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
tpu-info

# Install git
sudo apt-get update && sudo apt-get install git -y

```

### **2.2 Clone the Repository**

Please configure your github credentials on this machine to clone the repo. Use the account on which the access has been granted. 

```
git clone https://github.com/engineerbharath12/whisper-on-jax.git

```

### **2.3 Install Python Dependencies**

The installation process is now streamlined. All required packages and their correct versions are pinned in the `setup.py` file.

```bash
# 1. Install the correct JAX version for TPUs
# This step is crucial as it installs the TPU-specific libraries from Google\'s repository.
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# 2. Navigate into the project directory
cd whisper-jax-google/

# 3. Upgrade pip (recommended)
python3.11 -m pip install --upgrade pip

# 4. Install the project and all dependencies
# This single command will install torch, transformers, librosa, etc., at their tested versions.
pip install -e .
```

### **2.4 System & Audio Tools**

```
# Recommended kernel memory setting
sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"

# Install ffmpeg for audio conversion
sudo apt install ffmpeg -y

```

## **3. Running the Benchmark**

This project uses a central `config.yml` file to manage key parameters for the transcription pipelines, such as `dtype`, `batch_size`, attention implementation, and chunking strategy.

### **3.1 Optimizing Throughput with Audio Speed-Up**

A key feature of this repository is the ability to speed up audio during pre-processing (using FFmpeg) to increase throughput.

*   **Mechanism:** Audio is sped up (e.g., 2.2x), transcribed, and timestamps are adjusted.
*   **Configuration:** Controlled via `config.yml`:
    ```yaml
    common:
      speed_factor: 2.2  # Default recommended value
    ```
*   **Recommendation:** Extensive benchmarking has shown that **2.2x** is the optimal speed factor. It delivers a significant boost in RTFx (~40% for Turbo) while maintaining 100% transcription accuracy.
    *   **2.2x:** High performance, perfect accuracy.
    *   **2.5x:** Higher performance, minor artifacts.
    *   **3.0x:** Not recommended (hallucinations on short segments).

### **3.2 Running the Short Audio Benchmark**

For short audio files (under 30 seconds), use the `libri-short.py` script. The script automatically picks up the `speed_factor` from `config.yml`.

You can specify the model to benchmark by passing the `--model_id` argument:

```bash
# Run with the default model (openai/whisper-large-v3)
python3.11 benchmarks/libri-short.py

# Run with a different model, e.g., the turbo version
python3.11 benchmarks/libri-short.py --model_id "openai/whisper-large-v3-turbo"
```

### **3.3 Running the Long Audio Benchmark**

For long audio files, use the `libri-long.py` script. This script is optimized for long-form transcription and will process the audio in chunks. It also respects the `speed_factor` in `config.yml`.

You can also specify the model for the long audio benchmark:

```bash
# Run with the default model
python3.11 benchmarks/libri-long.py

# Run with the turbo model
python3.11 benchmarks/libri-long.py --model_id "openai/whisper-large-v3-turbo"
```

### **3.4 Running the Online Server Benchmark**

For production scenarios, you should benchmark the Online Serving Pipeline (`pipeline_online.py`), which simulates a real-world load with concurrent HTTP requests.

**1. Start the Server:**
```bash
cd server/
python3.11 -m uvicorn main:app --host 0.0.0.0 --port 8000
```
*Wait for the logs to show "Application startup complete" (this includes JIT compilation).*

**2. Run the "Smart" Benchmark Client:**
Open a new terminal window and run:
```bash
# Benchmark with 1280 concurrent requests
python3.11 benchmarks/test_api_smart.py -n 1280 -c 1280
```
*   `-n`: Total number of requests.
*   `-c`: Concurrency level (simultaneous clients).

### **3.5 Analyze Results**

For all scripts, the benchmark will first perform a one-time JIT compilation, which can take several minutes. After compilation, it will execute the transcription and print a final summary table with the performance metrics, including the Real-Time Factor (RTFx).

```
