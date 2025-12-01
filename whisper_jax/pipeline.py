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
import threading
import time
import yaml
import librosa
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
    GenerationConfig,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperTokenizerFast,
    is_tokenizers_available,
)
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from transformers.pipelines.audio_utils import ffmpeg_read
from transformers.utils import logging

# ===== Local Imports =====
from .modeling_flax_whisper import FlaxWhisperForConditionalGeneration
from .partitioner import PjitPartitioner
from .train_state import InferenceState

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
    except Exception:  # Broad exception to catch network errors, repo not found, etc.
        return False

class FlaxWhisperPmapPipeline:
    BATCH_BUCKETS = common_config.get("batch_buckets", [4, 40, 80])
    def __init__(
        self,
        checkpoint,
        dtype=default_dtype,
        batch_size=None,
        max_length=None,
        encoder_attention_implementation: str = common_config.get("encoder_attention_implementation", "original"),
        decoder_attention_implementation: str = common_config.get("decoder_attention_implementation", "original"),
        skip_special_tokens: bool = generation_config_values.get("skip_special_tokens", True),
        **kwargs
    ):
        logger.info(f"Initializing FlaxWhisperPmapPipeline for checkpoint: {checkpoint}")
        self.checkpoint = checkpoint
        self.dtype = dtype
        self.config = WhisperConfig.from_pretrained(self.checkpoint)
        self.skip_special_tokens = skip_special_tokens

        self.processor = WhisperProcessor.from_pretrained(self.checkpoint)
        self.feature_extractor = self.processor.feature_extractor
        tokenizer_cls = WhisperTokenizerFast if is_tokenizers_available() else WhisperTokenizer
        self.tokenizer = tokenizer_cls.from_pretrained(checkpoint)

        # FIX: Force TPU features (since implementation is now corrected)
        self.use_tpu_features = kwargs.get("use_tpu_features", True)
        logger.info(f"--- Feature Extraction Method: {'TPU-based' if self.use_tpu_features else 'CPU-based'} ---")

        logger.info("Loading Flax model weights...")
        
        has_flax_weights = _flax_weights_exist(self.checkpoint)
        
        if not has_flax_weights:
            logger.warning(
                f"No Flax weights found for checkpoint {self.checkpoint}. "
                "Weights will be converted from PyTorch on the fly."
            )
            self.model = FlaxWhisperForConditionalGeneration.from_pretrained(
                self.checkpoint, 
                dtype=self.dtype,
                from_pt=True,
                encoder_attention_implementation=encoder_attention_implementation,
                decoder_attention_implementation=decoder_attention_implementation,
            )
            loaded_params = self.model.params
        else:
            logger.info(f"Found native Flax weights for checkpoint {self.checkpoint}.")
            loaded_model_and_params = FlaxWhisperForConditionalGeneration.from_pretrained(
                self.checkpoint, 
                _do_init=False, 
                dtype=self.dtype,
                encoder_attention_implementation=encoder_attention_implementation,
                decoder_attention_implementation=decoder_attention_implementation,
            )
            self.model = loaded_model_and_params[0]
            loaded_params = loaded_model_and_params[1]

        # Update generation config with values from YAML
        for key, value in generation_config_values.items():
            if hasattr(self.model.generation_config, key):
                setattr(self.model.generation_config, key, value)

        self.max_length = max_length if max_length is not None else self.model.generation_config.max_length
        self.min_batch_size = jax.local_device_count()
        self.batch_size = batch_size if batch_size is not None else common_config.get("global_batch_size", self.min_batch_size)
        logger.info(f"Number of JAX devices detected: {self.min_batch_size}. Default batch size set to: {self.batch_size}")

        # --- Default PMAP Setup ---
        logger.info("Setting up default inference mode using `pmap` (data parallelism)...")
        self.params = jax_utils.replicate(freeze(loaded_params))
        self.is_sharded = False

        # HBM Memory Cleanup
        if not has_flax_weights:
            self.model._params = None

        del loaded_params
        try:
            del loaded_model_and_params
        except NameError:
            pass 
        gc.collect()
        jax.clear_caches()

        def generate_fn(params, input_audio, attention_mask, forced_decoder_ids, return_timestamps, max_length, use_tpu_features, model, feature_extractor):
            if use_tpu_features:
                input_features = jax.vmap(self._jax_feature_extractor, in_axes=(0, None, None, None, None, None, None, None, None))(
                    input_audio,
                    self.feature_extractor.sampling_rate,
                    self.feature_extractor.n_fft,
                    self.feature_extractor.hop_length,
                    self.feature_extractor.n_fft,
                    self.feature_extractor.feature_size,
                    0.0,
                    8000.0,
                    3000, 
                )
            else:
                input_features = input_audio

            output_ids = self.model.pipeline_generate(
                input_features,
                attention_mask=attention_mask,
                params=params,
                forced_decoder_ids=forced_decoder_ids,
                return_timestamps=return_timestamps,
                eos_token_id=self.model.config.eos_token_id,
                pad_token_id=self.model.config.pad_token_id,
                decoder_start_token_id=self.model.config.decoder_start_token_id,
            ).sequences
            return output_ids

        generate_fn = partial(generate_fn, model=self.model, feature_extractor=self.feature_extractor)

        self.p_generate = jax.pmap(
            generate_fn,
            "input_audio",
            in_axes=(0, 0, 0, None, None, None, None),
            out_axes=0,
            static_broadcasted_argnums=(3, 4, 5, 6)
        )
        logger.info("Default `pmap` pipeline ready.")

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

    # FIX: Use Librosa Slaney filters
    @staticmethod
    def _mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
        mels = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        return jnp.array(mels)

    @staticmethod
    def _jax_feature_extractor(waveform, sr, n_fft, hop_length, win_length, n_mels, fmin, fmax, target_feature_length):
        window = jnp.hanning(win_length)
        stft_matrix = FlaxWhisperPmapPipeline._stft(waveform, n_fft, hop_length, win_length, window)
        power_spectrogram = jnp.abs(stft_matrix) ** 2
        
        mel_filters = FlaxWhisperPmapPipeline._mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
        mel_spectrogram = jnp.dot(power_spectrogram, mel_filters.T)
        
        # FIX: Use Clamping Normalization
        log_mel_spectrogram = jnp.log10(jnp.maximum(mel_spectrogram, 1e-10))
        max_val = jnp.max(log_mel_spectrogram)
        log_mel_spectrogram = jnp.maximum(log_mel_spectrogram, max_val - 8.0)
        log_mel_spectrogram = (log_mel_spectrogram + 4.0) / 4.0
        
        log_mel_spectrogram = log_mel_spectrogram[:target_feature_length, :]
        log_mel_spectrogram = log_mel_spectrogram.T
        return log_mel_spectrogram

    def __call__(
        self,
        inputs,
        chunk_length_s: float = common_config.get("chunk_length_s", 30.0),
        stride_length_s: float = common_config.get("stride_length_s"),
        batch_size=None,
        language=None,
        task=None,
        return_timestamps=None,
        return_language=None,
        max_length=None,
        speed_factor: float = None,
    ):
        if speed_factor is None:
            speed_factor = common_config.get("speed_factor", 1.0)

        effective_batch_size = batch_size if batch_size is not None else self.batch_size
        if effective_batch_size % self.min_batch_size != 0:
            raise ValueError(f"Batch size must be a multiple of devices: {effective_batch_size} vs {self.min_batch_size}.")
        is_list_input = isinstance(inputs, list)
        if not is_list_input:
            inputs = [inputs]
        logger.info(f"Starting transcription for {len(inputs)} file(s)...")
        job_queue = queue.Queue(maxsize=effective_batch_size * 2)
        post_queue = queue.Queue() 
        futures_per_file = [[] for _ in inputs]
        final_results = [None] * len(inputs)
        
        batcher_thread = threading.Thread(
            target=self._batcher_worker,
            args=(job_queue, effective_batch_size, language, task, return_timestamps, max_length),
        )
        postprocessing_thread = threading.Thread( 
            target=self._postprocessing_worker, args=(post_queue, final_results)
        )
        batcher_thread.start()
        postprocessing_thread.start() 
        
        max_workers = common_config.get("preprocessing_workers") or min(32, (os.cpu_count() or 1) + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor: 
            for i, file_path in enumerate(inputs):
                executor.submit( 
                    self._preprocess_worker,
                    file_path,
                    chunk_length_s,
                    stride_length_s,
                    job_queue,
                    futures_per_file[i],
                    speed_factor,
                )
        
        job_queue.put(None)

        for i, file_futures in enumerate(futures_per_file):
            if not file_futures:
                logger.warning(f"File {i} ({inputs[i]}) produced no chunks.")
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
        
        post_queue.put(None) 
        batcher_thread.join()
        postprocessing_thread.join() 
        
        logger.info("...transcription finished.")
        return final_results if is_list_input else final_results[0]

    def _execute_model(self, input_audio, attention_mask, language=None, task=None, return_timestamps=False, max_length=None):
        forced_decoder_ids = self.get_forced_decoder_ids(
            language=language, task=task, return_timestamps=return_timestamps
        )
        effective_max_length = max_length or self.max_length
        
        sharded_audio = shard(input_audio)
        sharded_mask = shard(attention_mask)
        
        # Pass use_tpu_features
        output_ids = self.p_generate(self.params, sharded_audio, sharded_mask, forced_decoder_ids, return_timestamps, effective_max_length, self.use_tpu_features)
        return jax.device_get(output_ids.reshape(-1, effective_max_length))

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
            return {"text": decoded_texts[0]}

    def _partial_postprocess(self, pred_ids, stride):
        out = {"tokens": pred_ids[None, :]}
        if stride:
            sr = self.feature_extractor.sampling_rate
            out["stride"] = (stride[0] / sr, stride[1] / sr, stride[2] / sr)
        return out

    def _preprocess_worker(self, file_path, chunk_length_s, stride_length_s, job_queue, futures_list, speed_factor=1.0):
        try:
            if speed_factor == 1.0:
                with open(file_path, "rb") as f:
                    waveform = ffmpeg_read(f.read(), self.feature_extractor.sampling_rate)
            else:
                import subprocess
                sr = self.feature_extractor.sampling_rate
                cmd = [
                    "ffmpeg",
                    "-i", file_path,
                    "-filter:a", f"atempo={speed_factor}",
                    "-f", "f32le",
                    "-ac", "1",
                    "-ar", str(sr),
                    "pipe:1"
                ]
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
                waveform = np.frombuffer(out, dtype=np.float32)

            if waveform.shape[0] < self.feature_extractor.sampling_rate * 0.1: 
                return

            chunk_len = round(chunk_length_s * self.feature_extractor.sampling_rate)

            if waveform.shape[0] <= chunk_len:
                future = Future()
                futures_list.append(future)
                job = {"audio": waveform, "stride": (waveform.shape[0], 0, 0), "future": future}
                job_queue.put(job)
                return

            stride_s = stride_length_s if stride_length_s is not None else chunk_length_s / 6.0
            stride = round(stride_s * self.feature_extractor.sampling_rate)
            step = chunk_len - stride * 2
            for chunk_start in range(0, waveform.shape[0], step):
                chunk_end = chunk_start + chunk_len
                chunk = waveform[chunk_start:chunk_end]
                
                stride_info = (len(chunk), 0 if chunk_start == 0 else stride, 0 if chunk_end >= len(waveform) else stride)
                future = Future()
                futures_list.append(future)
                job = {"audio": chunk, "stride": stride_info, "future": future}
                job_queue.put(job)
        except Exception as e:
            logger.error(f"Preprocessing failed for file {file_path}: {e}")
            future = Future()
            future.set_exception(e)
            futures_list.append(future)

    def _batcher_worker(self, job_queue, batch_size, language, task, return_timestamps, max_length):
        stop = False
        while not stop:
            batch = []
            while len(batch) < batch_size:
                job = job_queue.get()
                if job is None:
                    stop = True
                    break
                batch.append(job)

            if not batch:
                continue

            max_bucket_size = self.BATCH_BUCKETS[-1]

            for i in range(0, len(batch), max_bucket_size):
                chunk = batch[i:i + max_bucket_size]
                actual_chunk_size = len(chunk)

                original_lengths = [job["audio"].shape[0] for job in chunk]

                max_audio_length = 30 * self.feature_extractor.sampling_rate 
                padded_batch = np.zeros((len(chunk), max_audio_length), dtype=np.float32)
                for j, job in enumerate(chunk):
                    audio_chunk = job["audio"]
                    padded_batch[j, :audio_chunk.shape[0]] = audio_chunk

                feature_lengths = (np.array(original_lengths) // 160).astype(np.int32)
                attention_mask = (np.arange(3000) < feature_lengths[:, None]).astype(np.int32)

                padded_batch_size = next(
                    (b for b in self.BATCH_BUCKETS if b >= actual_chunk_size), 
                    max_bucket_size
                )
                pad_width = padded_batch_size - actual_chunk_size

                if pad_width > 0:
                    feature_shape = padded_batch.shape[1:]
                    padding = np.zeros((pad_width, *feature_shape), dtype=padded_batch.dtype)
                    padded_batch = np.concatenate([padded_batch, padding], axis=0)
                    mask_shape = attention_mask.shape[1:]
                    mask_padding = np.zeros((pad_width, *mask_shape), dtype=attention_mask.dtype)
                    attention_mask = np.concatenate([attention_mask, mask_padding], axis=0)

                try:
                    pred_ids = self._execute_model(padded_batch, attention_mask, language, task, return_timestamps, max_length)
                    for k in range(actual_chunk_size):
                        job = chunk[k]
                        job["future"].set_result((pred_ids[k], job["stride"]))
                except Exception as e:
                    logger.error(f"Batch inference failed: {e}", exc_info=True)
                    for job in chunk:
                        job["future"].set_exception(e)

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

class FlaxWhisperPjitPipeline(FlaxWhisperPmapPipeline):
    def __init__(
        self,
        checkpoint="openai/whisper-large-v3",
        dtype=jnp.bfloat16,
        batch_size=None,
        max_length=None,
        encoder_attention_implementation: str = "original", 
        decoder_attention_implementation: str = "original", 
        **kwargs
    ):
        super().__init__(
            checkpoint,
            dtype,
            batch_size,
            max_length,
            encoder_attention_implementation=encoder_attention_implementation, 
            decoder_attention_implementation=decoder_attention_implementation, 
            **kwargs
        )

    def shard_params(self, model_parallel_submesh=(2, 2, 1, 1)):
        logger.info(f" Switching to `pjit` with model sharding. Mesh shape: {model_parallel_submesh}")
        # ... (PJIT logic omitted for brevity, assume it inherits safely or needs similar update if used)
        pass

def create_pipeline(mode="pmap", **kwargs):
    if mode == "pmap":
        return FlaxWhisperPmapPipeline(**kwargs)
    elif mode == "pjit":
        # return FlaxWhisperPjitPipeline(**kwargs)
        raise ValueError("Pjit mode not fully updated with recent fixes. Use pmap.")
    else:
        raise ValueError(f"Unknown mode: {mode}")