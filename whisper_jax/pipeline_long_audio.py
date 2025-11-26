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
import librosa  # Import librosa for Slaney filters
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
    try:
        files = list_repo_files(checkpoint)
        return "flax_model.msgpack" in files
    except Exception:
        return False

class FlaxWhisperPmapPipeline:
    BATCH_BUCKETS = common_config.get("batch_buckets", [4, 40, 80])

    def __init__(self, checkpoint, dtype=default_dtype, batch_size=None, max_length=None, skip_special_tokens: bool = generation_config_values.get("skip_special_tokens", True), **kwargs):
        logger.info(f"Initialising FlaxWhisperPmapPipeline for checkpoint: {checkpoint}")
        self.checkpoint = checkpoint
        self.dtype = dtype
        self.config = WhisperConfig.from_pretrained(self.checkpoint)
        self.processor = WhisperProcessor.from_pretrained(self.checkpoint)
        self.feature_extractor = self.processor.feature_extractor
        tokenizer_cls = WhisperTokenizerFast if is_tokenizers_available() else WhisperTokenizer
        self.tokenizer = tokenizer_cls.from_pretrained(checkpoint)
        self.skip_special_tokens = skip_special_tokens
        
        # --- FEATURE EXTRACTION FIX ---
        # Now that we have fixed the JAX implementation to match Hugging Face (Slaney + Clamping),
        # we can safely use TPU feature extraction for ALL models, including V3 Large.
        self.use_tpu_features = kwargs.get("use_tpu_features", True)
        logger.info(f"--- Feature Extraction Method: {'TPU-based' if self.use_tpu_features else 'CPU-based'} ---")
        
        has_flax_weights = _flax_weights_exist(self.checkpoint)
        
        if not has_flax_weights:
            self.model = FlaxWhisperForConditionalGeneration.from_pretrained(
                self.checkpoint, dtype=self.dtype, from_pt=True, **kwargs
            )
            loaded_params = self.model.params
        else:
            self.model, loaded_params = FlaxWhisperForConditionalGeneration.from_pretrained(
                self.checkpoint, dtype=self.dtype, _do_init=False, **kwargs
            )
        
        for key, value in generation_config_values.items():
            if hasattr(self.model.generation_config, key):
                setattr(self.model.generation_config, key, value)

        self.max_length = max_length if max_length is not None else self.model.generation_config.max_length
        self.min_batch_size = jax.local_device_count()
        self.batch_size = batch_size if batch_size is not None else common_config.get("global_batch_size", self.min_batch_size)

        self.params = jax_utils.replicate(freeze(loaded_params))
        
        del loaded_params
        gc.collect()
        jax.clear_caches()
        
        generate_fn = partial(self._generate_fn, model=self.model, feature_extractor=self.feature_extractor)
        
        self.p_generate = jax.pmap(
            generate_fn, 
            "input_audio", 
            in_axes=(0, 0, 0, None, None, None, None), 
            out_axes=0, 
            static_broadcasted_argnums=(3, 4, 5, 6)
        )
        logger.info("`pmap` pipeline ready.")

    def __call__(self, inputs, 
        chunk_length_s: float = common_config.get("chunk_length_s", 30.0),
        stride_length_s: float = common_config.get("stride_length_s"),
        batch_size=None, language=None, task=None, return_timestamps=None, return_language=None, max_length=None, 
        speed_factor: float = 1.0, # New param
        **kwargs):
        effective_batch_size = batch_size if batch_size is not None else self.batch_size
        if effective_batch_size % self.min_batch_size != 0:
            raise ValueError(f"Batch size must be a multiple of devices: {effective_batch_size} vs {self.min_batch_size}.")
        
        is_list_input = isinstance(inputs, list)
        if not is_list_input:
            inputs = [inputs]
        logger.info(f"Starting transcription for {len(inputs)} file(s)...")

        job_queue = queue.Queue(maxsize=effective_batch_size * 2)
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

        with ThreadPoolExecutor(max_workers=len(inputs)) as executor:
            for i, file_path in enumerate(inputs):
                executor.submit(self._file_processor_worker, file_path, chunk_length_s, stride_length_s, job_queue, post_queue, i, return_timestamps, return_language, speed_factor)

        job_queue.put(None)
        self.fetch_queue.put(None)
        post_queue.put(None)
        
        batcher_thread.join()
        result_fetcher_thread.join()
        postprocessing_thread.join()
        
        logger.info("...transcription finished.")
        return final_results if is_list_input else final_results[0]

    def _file_processor_worker(self, file_path, chunk_length_s, stride_length_s, job_queue, post_queue, file_index, return_timestamps, return_language, speed_factor):
        file_futures = []
        self._streaming_preprocess_worker(file_path, chunk_length_s, stride_length_s, job_queue, file_futures, speed_factor)
        
        wait(file_futures)
        unpacked_outputs = []
        try:
            for future in file_futures:
                pred_ids, stride_info = future.result()
                unpacked_outputs.append(self._partial_postprocess(pred_ids, stride_info))
            post_queue.put((file_index, unpacked_outputs, return_timestamps, return_language))
        except Exception as e:
            logger.error(f"Failed to retrieve results for file {file_index}: {e}")
            post_queue.put((file_index, e, return_timestamps, return_language))

    def _streaming_preprocess_worker(self, file_path, chunk_length_s, stride_length_s, job_queue, futures_list, speed_factor=1.0):
        try:
            command = ['ffmpeg', '-i', file_path]
            if speed_factor != 1.0:
                command.extend(['-filter:a', f'atempo={speed_factor}'])
            
            command.extend(['-f', 'f32le', '-ar', str(self.feature_extractor.sampling_rate), '-ac', '1', '-nostdin', 'pipe:1'])
            
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

            chunk_len = round(chunk_length_s * self.feature_extractor.sampling_rate)
            stride_s = stride_length_s if stride_length_s is not None else chunk_length_s / 6.0
            stride = round(stride_s * self.feature_extractor.sampling_rate)
            step = chunk_len - stride * 2

            in_bytes = process.stdout.read(chunk_len * 4)
            if not in_bytes:
                return

            waveform = np.frombuffer(in_bytes, dtype=np.float32)
            
            stride_info = (len(waveform), 0, stride if len(waveform) == chunk_len else 0)
            future = Future()
            futures_list.append(future)
            job = {"audio": waveform, "stride": stride_info, "future": future}
            job_queue.put(job)

            while True:
                next_in_bytes = process.stdout.read(step * 4)
                if not next_in_bytes:
                    break

                new_waveform_part = np.frombuffer(next_in_bytes, dtype=np.float32)
                waveform = np.concatenate([waveform[-stride*2:], new_waveform_part])

                if len(waveform) == 0:
                    break

                stride_info = (len(waveform), stride, stride if len(waveform) == chunk_len else 0)
                future = Future()
                futures_list.append(future)
                job = {"audio": waveform, "stride": stride_info, "future": future}
                job_queue.put(job)

        except Exception as e:
            logger.error(f"Streaming worker failed for {file_path}: {e}")

    def _batcher_worker(self, job_queue, batch_size, kwargs):
        # Use the new "Patient Batcher" logic
        initial_timeout = long_audio_config.get("batcher_initial_timeout_s", 5) # Longer initial wait
        assembly_timeout = long_audio_config.get("batch_assembly_timeout_s", 0.1) # Default to 0.2s if not set

        while True:
            batch_jobs = []
            try:
                # 1. Wait for the first item
                first_job = job_queue.get(timeout=initial_timeout)
                if first_job is None:
                    break # End of stream
                batch_jobs.append(first_job)

                # 2. Start the timer and patiently fill the batch
                assembly_deadline = time.time() + assembly_timeout
                while len(batch_jobs) < batch_size:
                    remaining_time = assembly_deadline - time.time()
                    if remaining_time <= 0:
                        break # Timer expired

                    try:
                        next_job = job_queue.get(timeout=remaining_time)
                        if next_job is None:
                            job_queue.put(None) # Put sentinel back for main loop
                            break
                        batch_jobs.append(next_job)
                    except queue.Empty:
                        break # Timer expired while waiting
            
            except queue.Empty:
                # This happens if the queue is empty for the initial_timeout, meaning all producers are done.
                break

            if not batch_jobs:
                continue

            # (The rest of the function for padding and execution remains the same)
            max_bucket_size = self.BATCH_BUCKETS[-1]

            for i in range(0, len(batch_jobs), max_bucket_size):
                chunk = batch_jobs[i:i + max_bucket_size]
                actual_chunk_size = len(chunk)

                if self.use_tpu_features:
                    padded_batch = np.zeros((len(chunk), 30 * self.feature_extractor.sampling_rate), dtype=np.float32)
                    for j, job in enumerate(chunk):
                        padded_batch[j, :job["audio"].shape[0]] = job["audio"]
                    attention_mask = None
                else:
                    input_features = self.processor(
                        [job["audio"] for job in chunk],
                        sampling_rate=self.feature_extractor.sampling_rate,
                        return_tensors="np",
                        padding="longest",
                    ).input_features
                    
                    padded_batch = np.pad(
                        input_features,
                        ((0, 0), (0, 0), (0, 3000 - input_features.shape[2])),
                        mode='constant'
                    )
                    attention_mask = None

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
            self.use_tpu_features
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
            text, optional = self.tokenizer._decode_asr(
                unpacked_outputs,
                return_timestamps=return_timestamps,
                return_language=return_language,
                time_precision=time_precision,
            )
            return {"text": text, **optional}
        else:
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
            input_features = jax.vmap(FlaxWhisperPmapPipeline._jax_feature_extractor, in_axes=(0, None, None, None, None, None, None, None, None))(
                input_data, feature_extractor.sampling_rate, feature_extractor.n_fft,
                feature_extractor.hop_length, feature_extractor.n_fft, feature_extractor.feature_size,
                0.0, 8000.0, 3000,
            )
        else:
            input_features = input_data

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
            padded_waveform, (win_length,), (hop_length,), 'VALID', dimension_numbers=('NWC', 'WIO', 'NWC')
        )[0]
        windowed_frames = frames * window
        stft_matrix = jnp.fft.rfft(windowed_frames, n=n_fft)
        return stft_matrix

    # FIX: Use Librosa filters for correctness (Slaney)
    @staticmethod
    def _mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
        # Calculate on CPU using librosa (Slaney norm)
        mels = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        return jnp.array(mels)

    @staticmethod
    def _jax_feature_extractor(waveform, sr, n_fft, hop_length, win_length, n_mels, fmin, fmax, target_feature_length):
        window = jnp.hanning(win_length)
        stft_matrix = FlaxWhisperPmapPipeline._stft(waveform, n_fft, hop_length, win_length, window)
        power_spectrogram = jnp.abs(stft_matrix) ** 2
        
        mel_filters = FlaxWhisperPmapPipeline._mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
        mel_spectrogram = jnp.dot(power_spectrogram, mel_filters.T)
        
        # FIX: Use Clamping Normalization (Whisper Standard) instead of Instance Norm
        log_mel_spectrogram = jnp.log10(jnp.maximum(mel_spectrogram, 1e-10))
        max_val = jnp.max(log_mel_spectrogram)
        log_mel_spectrogram = jnp.maximum(log_mel_spectrogram, max_val - 8.0)
        log_mel_spectrogram = (log_mel_spectrogram + 4.0) / 4.0
        
        log_mel_spectrogram = log_mel_spectrogram[:target_feature_length, :]
        log_mel_spectrogram = log_mel_spectrogram.T
        return log_mel_spectrogram

def create_pipeline(mode="pmap", **kwargs):
    if mode == "pmap":
        return FlaxWhisperPmapPipeline(**kwargs)
    else:
        raise ValueError(f"Mode {mode} not supported in this version.")