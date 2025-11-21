# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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

# ===== Standard Library Imports =====
import gc
import os
import queue
import re
import subprocess
import threading
import time
import yaml
from functools import partial
from concurrent.futures import Future, ThreadPoolExecutor, wait

# ===== Third-Party Library Imports =====
import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import list_repo_files
from jax import lax
from flax import jax_utils
from flax.core.frozen_dict import freeze
from flax.training.common_utils import shard
from jax.sharding import PartitionSpec as P
from transformers import (
    WhisperConfig,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperTokenizerFast,
    is_tokenizers_available,
)
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from transformers.utils import logging

# ===== Local Imports =====
from .modeling_flax_whisper import FlaxWhisperForConditionalGeneration

# ===== Globals =====
logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

# Load configuration from YAML file
try:
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yml")
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    logger.warning("config.yml not found, using default values.")
    config = {}

common_config = config.get("common", {})
generation_config_values = config.get("generation", {})
long_audio_config = config.get("long_audio", {})

# Map string dtype to jnp dtype
dtype_map = {
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
    "float32": jnp.float32,
}
default_dtype_str = common_config.get('dtype', 'bfloat16')
default_dtype = dtype_map.get(default_dtype_str, jnp.bfloat16)


def _flax_weights_exist(checkpoint):
    """
    Checks if a checkpoint has native Flax weights (`flax_model.msgpack`) available on the Hugging Face Hub.
    """
    try:
        files = list_repo_files(checkpoint)
        return "flax_model.msgpack" in files
    except Exception:  # Broad exception to catch network errors, repo not found, etc.
        return False

# ===== Standalone Helper Functions (Step 1) =====

def get_duration(file_path):
    """Gets the duration of an audio file using ffprobe."""
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", file_path
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        return float(result.stdout)
    except Exception as e:
        logger.error(f"Could not get duration for {file_path}: {e}")
        return 0

def ffmpeg_worker(task_queue, job_queue, sampling_rate):
    """
    A worker THREAD that reads a specific time chunk from a file using ffmpeg.
    """
    while True:
        item = task_queue.get()
        if item is None:
            task_queue.put(None) # Propagate the sentinel for other workers
            break
        
        file_path, chunk_index, start_time, duration, stride_info, future = item
        try:
            command = [
                'ffmpeg', '-ss', str(start_time), '-t', str(duration),
                '-i', file_path, '-f', 'f32le', '-ar', str(sampling_rate), '-ac', '1', '-'
            ]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            stdout, _ = process.communicate()
            waveform = np.frombuffer(stdout, dtype=np.float32)
            
            if waveform.shape[0] > 0:
                job_queue.put({"audio": waveform, "stride": stride_info, "future": future})
            else:
                # This can happen for very short chunks at the end of the file
                future.set_result((np.array([], dtype=np.int32), stride_info))
                
        except Exception as e:
            logger.error(f"ffmpeg worker failed for chunk {chunk_index}: {e}")
            future.set_exception(e)

# ===== Unified FlaxWhisperPmapPipeline Class (Step 2) =====

class FlaxWhisperPmapPipeline:
    BATCH_BUCKETS = common_config.get("batch_buckets", [4, 40, 80])

    def __init__(self, checkpoint, dtype=default_dtype, batch_size=None, max_length=None, skip_special_tokens: bool = generation_config_values.get("skip_special_tokens", True), **kwargs):
        logger.info(f"🚀 Initializing FlaxWhisperPmapPipeline for checkpoint: {checkpoint}")
        self.checkpoint = checkpoint
        self.dtype = dtype
        self.config = WhisperConfig.from_pretrained(self.checkpoint)
        self.processor = WhisperProcessor.from_pretrained(self.checkpoint)
        self.feature_extractor = self.processor.feature_extractor
        tokenizer_cls = WhisperTokenizerFast if is_tokenizers_available() else WhisperTokenizer
        self.tokenizer = tokenizer_cls.from_pretrained(checkpoint)
        self.skip_special_tokens = skip_special_tokens
        
        self.use_tpu_features = self.checkpoint != "openai/whisper-large-v3-turbo"
        logger.info(f"--- 💡 Feature Extraction Method: {'TPU-based' if self.use_tpu_features else 'CPU-based'} ---")
        
        has_flax_weights = _flax_weights_exist(self.checkpoint)
        
        if not has_flax_weights:
            logger.warning(
                f"No Flax weights found for checkpoint {self.checkpoint}. "
                "Weights will be converted from PyTorch on the fly."
            )
            self.model = FlaxWhisperForConditionalGeneration.from_pretrained(
                self.checkpoint, dtype=self.dtype, from_pt=True, **kwargs
            )
            loaded_params = self.model.params
        else:
            self.model, loaded_params = FlaxWhisperForConditionalGeneration.from_pretrained(
                self.checkpoint, dtype=self.dtype, _do_init=False, **kwargs
            )
        
        # Update generation config with values from YAML
        for key, value in generation_config_values.items():
            if hasattr(self.model.generation_config, key):
                setattr(self.model.generation_config, key, value)

        self.max_length = max_length if max_length is not None else self.model.generation_config.max_length
        self.min_batch_size = jax.local_device_count()
        self.batch_size = batch_size if batch_size is not None else common_config.get("global_batch_size", self.min_batch_size)

        # PMAP-specific setup
        self.params = jax_utils.replicate(freeze(loaded_params))
        
        if not has_flax_weights:
            self.model._params = None
        
        # Manually free memory of the original CPU-loaded params
        del loaded_params
        gc.collect()
        jax.clear_caches()
        
        # Create a partial function with model baked in
        generate_fn = partial(self._generate_fn, model=self.model, feature_extractor=self.feature_extractor)
        
        self.p_generate = jax.pmap(
            generate_fn, 
            "input_audio", 
            in_axes=(0, 0, 0, None, None, None, None), 
            out_axes=0, 
            static_broadcasted_argnums=(3, 4, 5, 6)
        )
        logger.info("✅ `pmap` pipeline ready.")

    def __call__(self, inputs, 
        # The length of audio chunks in seconds. Default is 30.0, matching the model's training.
        chunk_length_s: float = common_config.get("chunk_length_s", 30.0),
        # The stride between audio chunks in seconds (e.g., an integer like 2, 4, 6). If None, defaults to chunk_length_s / 6.0.
        stride_length_s: float = common_config.get("stride_length_s"),
        batch_size=None, language=None, task=None, return_timestamps=None, return_language=None, max_length=None, **kwargs):
        effective_batch_size = batch_size if batch_size is not None else self.batch_size
        if effective_batch_size % self.min_batch_size != 0:
            raise ValueError(f"Batch size must be a multiple of devices: {effective_batch_size} vs {self.min_batch_size}.")
        
        is_list_input = isinstance(inputs, list)
        if not is_list_input:
            inputs = [inputs]
        logger.info(f"Starting transcription for {len(inputs)} file(s)...")

        # --- STAGE 1: Create a "Blueprint" of All Work ---
        ffmpeg_task_queue = queue.Queue()
        futures_per_file = [[] for _ in inputs]

        with ThreadPoolExecutor() as executor:
            duration_futures = [executor.submit(get_duration, file_path) for file_path in inputs]
            durations = [future.result() for future in duration_futures]

        for i, (file_path, file_duration) in enumerate(zip(inputs, durations)):
            if file_duration <= 0:
                logger.warning(f"Could not get duration or file is empty: {file_path}. Skipping.")
                continue

            if file_duration <= chunk_length_s:
                # Process as a single chunk
                future = Future()
                futures_per_file[i].append(future)
                stride_info = (round(file_duration * self.feature_extractor.sampling_rate), 0, 0)
                ffmpeg_task = (file_path, 0, 0.0, file_duration, stride_info, future)
                ffmpeg_task_queue.put(ffmpeg_task)
            else:
                # Use sliding window for long files
                effective_stride_s = stride_length_s if stride_length_s is not None else chunk_length_s / 6.0
                chunk_len = round(chunk_length_s * self.feature_extractor.sampling_rate)
                stride = round(effective_stride_s * self.feature_extractor.sampling_rate)
                
                for chunk_start in range(0, int(file_duration * self.feature_extractor.sampling_rate), chunk_len - stride):
                    chunk_end = chunk_start + chunk_len
                    current_chunk_duration = (chunk_end - chunk_start) / self.feature_extractor.sampling_rate

                    if current_chunk_duration <= 0: continue

                    future = Future()
                    futures_per_file[i].append(future)

                    stride_info = (chunk_len, stride, stride)
                    
                    ffmpeg_task = (file_path, 0, chunk_start / self.feature_extractor.sampling_rate, current_chunk_duration, stride_info, future)
                    ffmpeg_task_queue.put(ffmpeg_task)

        ffmpeg_task_queue.put(None) # Sentinel for ffmpeg workers

        # --- STAGE 2: Start Parallel ffmpeg Worker Threads ---
        job_queue = queue.Queue(maxsize=effective_batch_size * 2)
        
        queue_size = ffmpeg_task_queue.qsize()
        num_ffmpeg_workers = common_config.get("preprocessing_workers") or min(32, (os.cpu_count() or 1) + 4, queue_size if queue_size > 1 else 2)
        num_ffmpeg_workers = max(1, num_ffmpeg_workers)

        ffmpeg_threads = []
        for _ in range(num_ffmpeg_workers):
            thread = threading.Thread(target=ffmpeg_worker, args=(ffmpeg_task_queue, job_queue, self.feature_extractor.sampling_rate))
            thread.start()
            ffmpeg_threads.append(thread)

        # --- STAGES 3 & 4: CONSUMER BACKEND & ASYNC POST-PROCESSING ---
        post_queue = queue.Queue()
        self.fetch_queue = queue.Queue()
        final_results = [None] * len(inputs)
        
        batcher_thread = threading.Thread(
            target=self._batcher_worker,
            args=(job_queue, effective_batch_size, {"language": language, "task": task, "return_timestamps": return_timestamps, "max_length": max_length}),
        )
        result_fetcher_thread = threading.Thread(target=self._result_fetcher_worker)
        postprocessing_thread = threading.Thread(
            target=self._postprocessing_worker, args=(post_queue, final_results)
        )
        batcher_thread.start()
        result_fetcher_thread.start()
        postprocessing_thread.start()
        
        # --- Final Collation with "Tag and Sort" ---
        for i, file_futures in enumerate(futures_per_file):
            if not file_futures:
                # This case is handled by the duration check at the beginning, but as a safeguard:
                post_queue.put((i, [], return_timestamps, return_language))
                continue
            
            wait(file_futures)
            
            unpacked_outputs = []
            try:
                for future in file_futures:
                    pred_ids, stride_info = future.result()
                    unpacked_outputs.append(self._partial_postprocess(pred_ids, stride_info))
                post_queue.put((i, unpacked_outputs, return_timestamps, return_language))
            except Exception as e:
                logger.error(f"Failed to retrieve results for file {i}: {e}")
                post_queue.put((i, e, return_timestamps, return_language))

        # --- FINAL CLEANUP ---
        job_queue.put(None) # Sentinel for batcher worker
        self.fetch_queue.put(None) # Sentinel for result fetcher worker
        post_queue.put(None) # Sentinel for post-processing worker
        
        for thread in ffmpeg_threads:
            thread.join()
        batcher_thread.join()
        result_fetcher_thread.join()
        postprocessing_thread.join()
        
        logger.info("...transcription finished.")
        return final_results if is_list_input else final_results[0]

    def _batcher_worker(self, job_queue, batch_size, kwargs):
        """The responsive batcher with a configurable timeout."""
        initial_timeout = long_audio_config.get("batcher_initial_timeout_s", 0.1)
        assembly_timeout = long_audio_config.get("batch_assembly_timeout_s")

        while True:
            batch_jobs = []
            try:
                # Wait for the first item, but only for a short time
                job = job_queue.get(timeout=initial_timeout)
                if job is None:
                    job_queue.put(None); break
                batch_jobs.append(job)
                
                # Greedily get the rest of a full batch without blocking if no assembly timeout is set
                if assembly_timeout is None:
                    while len(batch_jobs) < batch_size:
                        job = job_queue.get_nowait()
                        if job is None:
                            job_queue.put(None); break
                        batch_jobs.append(job)
                else:
                    # Patiently wait for more jobs to arrive to form a fuller batch
                    assembly_deadline = time.time() + assembly_timeout
                    while len(batch_jobs) < batch_size and time.time() < assembly_deadline:
                        remaining_time = assembly_deadline - time.time()
                        if remaining_time <= 0: break
                        try:
                            job = job_queue.get(timeout=remaining_time)
                            if job is None:
                                job_queue.put(None); break
                            batch_jobs.append(job)
                        except queue.Empty:
                            break # Assembly timeout hit
            except queue.Empty:
                pass # This is expected when the queue is temporarily empty

            if not batch_jobs:
                continue

            # The rest of the batching logic for padding and calling _execute_model
            max_bucket_size = self.BATCH_BUCKETS[-1]

            for i in range(0, len(batch_jobs), max_bucket_size):
                chunk = batch_jobs[i:i + max_bucket_size]
                actual_chunk_size = len(chunk)

                # --- Conditional Feature Extraction ---
                if self.use_tpu_features:
                    logger.info("--- 💡 VERIFICATION: Using TPU Feature Extraction Path ---")
                    # Pad raw audio for on-device feature extraction
                    padded_batch = np.zeros((len(chunk), 30 * self.feature_extractor.sampling_rate), dtype=np.float32)
                    for j, job in enumerate(chunk):
                        padded_batch[j, :job["audio"].shape[0]] = job["audio"]
                    attention_mask = None # Not needed for on-device extraction
                else:
                    # Pre-process on CPU using the transformers processor
                    input_features = self.processor(
                        [job["audio"] for job in chunk],
                        sampling_rate=self.feature_extractor.sampling_rate,
                        return_tensors="np",
                        padding="longest", # Pad to the longest in the batch
                    ).input_features
                    
                    # Manually pad the features to the required 3000 length
                    padded_batch = np.pad(
                        input_features,
                        ((0, 0), (0, 0), (0, 3000 - input_features.shape[2])),
                        mode='constant'
                    )
                    attention_mask = None # Not needed for this path either

                padded_batch_size = next(
                    (b for b in self.BATCH_BUCKETS if b >= actual_chunk_size), 
                    max_bucket_size
                )
                pad_width = padded_batch_size - actual_chunk_size

                if pad_width > 0:
                    feature_shape = padded_batch.shape[1:]
                    padding = np.zeros((pad_width, *feature_shape), dtype=padded_batch.dtype)
                    padded_batch = np.concatenate([padded_batch, padding], axis=0)
                    if attention_mask is not None:
                        attention_mask = np.concatenate([attention_mask, np.zeros((pad_width, *attention_mask.shape[1:]), dtype=attention_mask.dtype)], axis=0)
                try:
                    pred_ids_on_device = self._execute_model(padded_batch, attention_mask, **kwargs)
                    self.fetch_queue.put((pred_ids_on_device, chunk, actual_chunk_size))
                except Exception as e:
                    logger.error(f"Batch inference failed: {e}", exc_info=True)
                    for job in chunk:
                        job["future"].set_exception(e)

    def _execute_model(self, input_data, attention_mask, language=None, task=None, return_timestamps=False, max_length=None):
        forced_decoder_ids = self.get_forced_decoder_ids(
            language=language, task=task, return_timestamps=return_timestamps
        )
        effective_max_length = max_length or self.max_length
        
        sharded_data = shard(input_data)
        sharded_mask = shard(attention_mask) if attention_mask is not None else None
        
        output_ids = self.p_generate(
            self.params,
            sharded_data,
            sharded_mask,
            forced_decoder_ids,
            return_timestamps,
            effective_max_length,
            self.use_tpu_features  # Pass the static flag
        )
        return output_ids.reshape(-1, effective_max_length)

    def get_forced_decoder_ids(self, task=None, language=None, return_timestamps=False):
        g = self.model.generation_config
        forced_decoder_ids = []
        if hasattr(g, "lang_to_id") and g.lang_to_id is not None:
            if language is not None:
                lang = language.lower()
                token_str = TO_LANGUAGE_CODE.get(lang, lang)
                token_key = f"<|{token_str}|>"
                if token_key in g.lang_to_id:
                    forced_decoder_ids.append((1, g.lang_to_id[token_key]))
                else:
                    raise ValueError(f"Unsupported language: {language}")
            task = task if task is not None else "transcribe"
            forced_decoder_ids.append((2, g.task_to_id[task]))
        if not return_timestamps:
            idx = len(forced_decoder_ids) + 1
            forced_decoder_ids.append((idx, g.no_timestamps_token_id))
        return tuple(forced_decoder_ids)

    def _final_decode(self, unpacked_outputs, return_timestamps, return_language):
        time_precision = self.feature_extractor.chunk_length / self.model.config.max_source_positions

        if return_timestamps:
            # When timestamps are returned, _decode_asr is used, which does not
            # use the skip_special_tokens parameter.
            text, optional = self.tokenizer._decode_asr(
                unpacked_outputs,
                return_timestamps=return_timestamps,
                return_language=return_language,
                time_precision=time_precision,
            )
            return {"text": text, **optional}
        else:
            # When timestamps are NOT returned, batch_decode is used with the
            # skip_special_tokens parameter.
            batch_token_ids = [output_dict["tokens"].flatten().tolist() for output_dict in unpacked_outputs]

            decoded_texts = self.processor.batch_decode(batch_token_ids, skip_special_tokens=self.skip_special_tokens)
            
            return {"text": " ".join(decoded_texts)}

    def _partial_postprocess(self, pred_ids, stride):
        out = {"tokens": pred_ids[None, :]}
        if stride:
            sr = self.feature_extractor.sampling_rate
            out["stride"] = (stride[0] / sr, stride[1] / sr, stride[2] / sr)
        return out

    def _postprocessing_worker(self, post_queue, final_results):
        while True:
            item = post_queue.get()
            if item is None: break
            file_index, sorted_results, return_timestamps, return_language = item
            try:
                if isinstance(sorted_results, Exception): raise sorted_results
                final_results[file_index] = self._final_decode(sorted_results, return_timestamps, return_language)
            except Exception as e:
                logger.error(f"Post-processing failed for file {file_index}: {e}")
                final_results[file_index] = {"text": f"[ERROR: {e}]"}

    def _result_fetcher_worker(self):
        while True:
            item = self.fetch_queue.get()
            if item is None:
                break
            
            pred_ids_on_device, chunk, actual_chunk_size = item
            
            try:
                pred_ids_cpu = jax.device_get(pred_ids_on_device)
                
                for i in range(actual_chunk_size):
                    job = chunk[i]
                    job["future"].set_result((pred_ids_cpu[i], job["stride"]))
            except Exception as e:
                logger.error(f"Result fetching failed: {e}", exc_info=True)
                for i in range(actual_chunk_size):
                    job = chunk[i]
                    job["future"].set_exception(e)

    @staticmethod
    def _generate_fn(params, input_data, attention_mask, forced_decoder_ids, return_timestamps, max_length, use_tpu_features, model, feature_extractor):
        if use_tpu_features:
            # On-device feature extraction for non-turbo models
            input_features = jax.vmap(FlaxWhisperPmapPipeline._jax_feature_extractor, in_axes=(0, None, None, None, None, None, None, None, None))(
                input_data, feature_extractor.sampling_rate, feature_extractor.n_fft,
                feature_extractor.hop_length, feature_extractor.n_fft, feature_extractor.feature_size,
                0.0, 8000.0, 3000,
            )
        else:
            # For turbo models, the features are pre-computed on CPU and passed in input_data
            input_features = input_data

        # Generate token ids
        output_ids = model.pipeline_generate(
            input_features, attention_mask=attention_mask, params=params,
            forced_decoder_ids=forced_decoder_ids, return_timestamps=return_timestamps,
            max_length=max_length,
        ).sequences
        return output_ids

    @staticmethod
    def _stft(waveform, n_fft, hop_length, win_length, window):
        padding = (win_length // 2, win_length // 2)
        padded_waveform = jnp.pad(waveform, padding, mode="reflect")
        padded_waveform = padded_waveform.reshape(1, -1, 1)
        frames = jax.lax.conv_general_dilated_patches(
            padded_waveform,
            (win_length,),
            (hop_length,),
            'VALID',
            dimension_numbers=('NWC', 'WIO', 'NWC')
        )[0]
        windowed_frames = frames * window
        stft_matrix = jnp.fft.rfft(windowed_frames, n=n_fft)
        return stft_matrix

    @staticmethod
    def _mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
        def hz_to_mel(freq):
            return 2595.0 * jnp.log10(1.0 + freq / 700.0)
        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
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

    @staticmethod
    def _jax_feature_extractor(waveform, sr, n_fft, hop_length, win_length, n_mels, fmin, fmax, target_feature_length):
        window = jnp.hanning(win_length)
        stft_matrix = FlaxWhisperPmapPipeline._stft(waveform, n_fft, hop_length, win_length, window)
        power_spectrogram = jnp.abs(stft_matrix) ** 2
        mel_filters = FlaxWhisperPmapPipeline._mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
        mel_spectrogram = jnp.dot(power_spectrogram, mel_filters.T)
        log_mel_spectrogram = jnp.log10(jnp.maximum(mel_spectrogram, 1e-10))
        log_mel_spectrogram = (log_mel_spectrogram - jnp.mean(log_mel_spectrogram)) / jnp.sqrt(jnp.var(log_mel_spectrogram) + 1e-7)
        log_mel_spectrogram = log_mel_spectrogram[:target_feature_length, :]
        log_mel_spectrogram = log_mel_spectrogram.T
        return log_mel_spectrogram

# ===== Factory Function (Step 3) =====

def create_pipeline(mode="pmap", **kwargs):
    if mode == "pmap":
        return FlaxWhisperPmapPipeline(**kwargs)
    else:
        raise ValueError(f"Mode {mode} not supported in this version.")