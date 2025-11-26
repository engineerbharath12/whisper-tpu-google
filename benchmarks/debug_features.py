import os
import numpy as np
import jax
import jax.numpy as jnp
import librosa
import subprocess
import tempfile
from transformers import WhisperProcessor

# --- JAX Feature Extractor Implementation ---
def _stft(waveform, n_fft, hop_length, win_length, window):
    padding = (win_length // 2, win_length // 2)
    padded_waveform = jnp.pad(waveform, padding, mode="reflect")
    padded_waveform = padded_waveform.reshape(1, -1, 1)
    frames = jax.lax.conv_general_dilated_patches(
        padded_waveform, (win_length,), (hop_length,), 'VALID', dimension_numbers=('NWC', 'WIO', 'NWC')
    )[0]
    windowed_frames = frames * window
    stft_matrix = jnp.fft.rfft(windowed_frames, n=n_fft)
    return stft_matrix

def _mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    def hz_to_mel(freq): return 2595.0 * jnp.log10(1.0 + freq / 700.0)
    def mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = jnp.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    fft_freqs = jnp.linspace(0, sr / 2, n_fft // 2 + 1)
    ramps = hz_points[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / (hz_points[1:-1, None] - hz_points[:-2, None])
    upper = ramps[2:] / (hz_points[2:, None] - hz_points[1:-1, None])
    fb = jnp.maximum(jnp.zeros_like(lower), jnp.minimum(lower, upper))
    return fb

def _jax_feature_extractor(waveform, sr, n_fft, hop_length, win_length, n_mels, fmin, fmax, target_feature_length):
    window = jnp.hanning(win_length)
    stft_matrix = _stft(waveform, n_fft, hop_length, win_length, window)
    power_spectrogram = jnp.abs(stft_matrix) ** 2
    mel_filters = _mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    mel_spectrogram = jnp.dot(power_spectrogram, mel_filters.T)
    log_mel_spectrogram = jnp.log10(jnp.maximum(mel_spectrogram, 1e-10))
    
    # Whisper Normalization: Clamp to top 80dB
    max_val = jnp.max(log_mel_spectrogram)
    log_mel_spectrogram = jnp.maximum(log_mel_spectrogram, max_val - 8.0)
    
    # Scale to [-1, 1] range (approx)
    log_mel_spectrogram = (log_mel_spectrogram + 4.0) / 4.0
    
    log_mel_spectrogram = log_mel_spectrogram[:target_feature_length, :]
    log_mel_spectrogram = log_mel_spectrogram.T
    return log_mel_spectrogram

# --- Helper to Speed Up Audio ---
def get_audio_data(file_path, speed=1.0):
    if speed == 1.0:
        y, sr = librosa.load(file_path, sr=16000)
        return y
    
    cmd = ["ffmpeg", "-i", file_path, "-filter:a", f"atempo={speed}", "-f", "wav", "-ar", "16000", "pipe:1"]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    import io
    import soundfile as sf
    y, sr = sf.read(io.BytesIO(out))
    return y.astype(np.float32)

# --- Comparison Logic ---
def compare_features(model_id, audio_path, speed=1.0):
    print(f"\n--- Comparing Features for {model_id} (Speed {speed}x) ---")
    
    processor = WhisperProcessor.from_pretrained(model_id)
    raw_audio = get_audio_data(audio_path, speed)
    print(f"Audio Shape: {raw_audio.shape}")
    
    # CPU Features
    cpu_inputs = processor(raw_audio, sampling_rate=16000, return_tensors="np")
    cpu_features = cpu_inputs.input_features[0]
    
    # TPU Features (JAX)
    target_len = 480000
    if len(raw_audio) < target_len:
        padded_audio = np.pad(raw_audio, (0, target_len - len(raw_audio)))
    else:
        padded_audio = raw_audio[:target_len]
        
    # Explicitly extract args
    n_fft = processor.feature_extractor.n_fft
    hop_length = processor.feature_extractor.hop_length
    n_mels = processor.feature_extractor.feature_size
    
    print(f"Params: n_fft={n_fft}, hop={hop_length}, n_mels={n_mels}")

    # Calling with explicit variables to avoid counting errors
    jax_features = _jax_feature_extractor(
        jnp.array(padded_audio), 
        16000, 
        n_fft, 
        hop_length, 
        n_fft, # win_length
        n_mels, 
        0.0, 
        8000.0, 
        3000
    )
    
    jax_features_np = np.array(jax_features)
    
    diff = np.abs(cpu_features - jax_features_np)
    mse = np.mean(diff ** 2)
    max_diff = np.max(diff)
    
    print(f"CPU Shape: {cpu_features.shape}")
    print(f"TPU Shape: {jax_features_np.shape}")
    print(f"MSE: {mse:.6f}")
    print(f"Max Diff: {max_diff:.6f}")
    
    if mse > 0.01: # Relaxed tolerance slightly for float diffs
        print("❌ SIGNIFICANT MISMATCH")
    else:
        print("✅ Match (within tolerance)")

if __name__ == "__main__":
    TEST_FILE = "/home/brathinam_google_com/14Oct/whisper-on-jax/asr_audio_new/18s/medical_domain_test.wav"
    if not os.path.exists(TEST_FILE):
        # Dummy generation if file missing
        sr = 16000
        t = np.linspace(0, 5, 5*sr)
        y = 0.5 * np.sin(2 * np.pi * 440 * t)
        import soundfile as sf
        sf.write("dummy_test.wav", y, sr)
        TEST_FILE = "dummy_test.wav"

    compare_features("openai/whisper-large-v3", TEST_FILE, speed=1.0)
    compare_features("openai/whisper-large-v3", TEST_FILE, speed=2.0)