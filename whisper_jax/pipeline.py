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
    """
    Checks if a checkpoint has native Flax weights (`flax_model.msgpack`) available on the Hugging Face Hub.
    """
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

        logger.info("Loading Flax model weights...")
        
        has_flax_weights = _flax_weights_exist(self.checkpoint)
        
        if not has_flax_weights:
            logger.warning(
                f"No Flax weights found for checkpoint {self.checkpoint}. "
                "Weights will be converted from PyTorch on the fly."
            )
            # from_pretrained with from_pt=True returns a single, fully-loaded model object
            self.model = FlaxWhisperForConditionalGeneration.from_pretrained(
                self.checkpoint, 
                dtype=self.dtype,
                from_pt=True,
                encoder_attention_implementation=encoder_attention_implementation,
                decoder_attention_implementation=decoder_attention_implementation,
            )
            # The parameters are an attribute of the loaded model
            loaded_params = self.model.params
        else:
            logger.info(f"Found native Flax weights for checkpoint {self.checkpoint}.")
            # For native Flax, we load the structure and weights separately.
            # This call returns a tuple (model, params)
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

        # For models converted from PyTorch, `self.model` still holds the unreplicated params.
        # We clear this reference to free up HBM on device 0.
        if not has_flax_weights:
            self.model._params = None

        # Manually free memory
        del loaded_params
        try:
            del loaded_model_and_params
        except NameError:
            pass  # This variable is not defined when loading from PyTorch, so we can safely ignore.
        gc.collect()
        jax.clear_caches()

        def generate_fn(params, input_audio, attention_mask, forced_decoder_ids, return_timestamps, max_length):
            # on-device feature extraction
            input_features = jax.vmap(self._jax_feature_extractor, in_axes=(0, None, None, None, None, None, None, None, None))(
                input_audio,
                self.feature_extractor.sampling_rate,
                self.feature_extractor.n_fft,
                self.feature_extractor.hop_length,
                self.feature_extractor.n_fft,
                self.feature_extractor.feature_size,
                0.0,
                8000.0,
                3000,  # target_feature_length
            )

            generation_config = self.model.generation_config
            generation_config.max_length = max_length

            output_ids = self.model.pipeline_generate(
                input_features,
                attention_mask=attention_mask,
                params=params,
                forced_decoder_ids=forced_decoder_ids,
                return_timestamps=return_timestamps,
                eos_token_id=self.model.config.eos_token_id,
                pad_token_id=self.model.config.pad_token_id,
                decoder_start_token_id=self.model.config.decoder_start_token_id,
                #generation_config=generation_config,
            ).sequences
            return output_ids

        self.p_generate = jax.pmap(
            generate_fn,
            "input_audio",
            in_axes=(0, 0, 0, None, None, None),
            out_axes=0,
            static_broadcasted_argnums=(3, 4, 5)
        )
        logger.info("Default `pmap` pipeline ready.")

    @staticmethod
    def _stft(waveform, n_fft, hop_length, win_length, window):
        """
        JAX implementation of Short-Time Fourier Transform.
        """
        # Pad the waveform to be a multiple of hop_length
        padding = (win_length // 2, win_length // 2)
        padded_waveform = jnp.pad(waveform, padding, mode="reflect")

        # Reshape to (N, W, C)
        padded_waveform = padded_waveform.reshape(1, -1, 1)

        # Frame the waveform
        frames = jax.lax.conv_general_dilated_patches(
            padded_waveform,
            (win_length,),
            (hop_length,),
            'VALID',
            dimension_numbers=('NWC', 'WIO', 'NWC')
        )[0]

        # Apply the window function
        windowed_frames = frames * window

        # Compute the STFT
        stft_matrix = jnp.fft.rfft(windowed_frames, n=n_fft)
        return stft_matrix

    @staticmethod
    def _mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
        """
        JAX implementation of Mel filterbank generation.
        """
        # Mel scale conversion
        def hz_to_mel(freq):
            return 2595.0 * jnp.log10(1.0 + freq / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        # Mel filterbank construction
        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = jnp.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        fft_freqs = jnp.linspace(0, sr / 2, n_fft // 2 + 1)

        # Slopes for the triangular filters
        ramps = hz_points[:, None] - fft_freqs[None, :]
        lower = -ramps[:-2] / (hz_points[1:-1, None] - hz_points[:-2, None])
        upper = ramps[2:] / (hz_points[2:, None] - hz_points[1:-1, None])
        
        # Combine the slopes to form the triangular filters
        fb = jnp.maximum(jnp.zeros_like(lower), jnp.minimum(lower, upper))
        return fb

    @staticmethod
    def _jax_feature_extractor(waveform, sr, n_fft, hop_length, win_length, n_mels, fmin, fmax, target_feature_length):
        """
        JAX implementation of the Whisper feature extractor.
        """
        # Create the window
        window = jnp.hanning(win_length)

        # Compute the STFT
        stft_matrix = FlaxWhisperPmapPipeline._stft(waveform, n_fft, hop_length, win_length, window)

        # Compute the power spectrogram
        power_spectrogram = jnp.abs(stft_matrix) ** 2

        # Compute the Mel filterbank
        mel_filters = FlaxWhisperPmapPipeline._mel_filterbank(sr, n_fft, n_mels, fmin, fmax)

        # Apply the Mel filterbank
        mel_spectrogram = jnp.dot(power_spectrogram, mel_filters.T)

        # Compute the log-Mel spectrogram
        log_mel_spectrogram = jnp.log10(jnp.maximum(mel_spectrogram, 1e-10))

        # Normalize the log-Mel spectrogram
        log_mel_spectrogram = (log_mel_spectrogram - jnp.mean(log_mel_spectrogram)) / jnp.sqrt(jnp.var(log_mel_spectrogram) + 1e-7)

        # Truncate to the required length (3000 frames)
        log_mel_spectrogram = log_mel_spectrogram[:target_feature_length, :]

        # Transpose to (num_mel_bins, num_frames)
        log_mel_spectrogram = log_mel_spectrogram.T

        return log_mel_spectrogram

    def __call__(
        self,
        inputs,
        # The length of audio chunks in seconds. Default is 30.0, matching the model's training.
        chunk_length_s: float = common_config.get("chunk_length_s", 30.0),
        # The stride between audio chunks in seconds (e.g., an integer like 2, 4, 6). If None, defaults to chunk_length_s / 6.0.
        stride_length_s: float = common_config.get("stride_length_s"),
        batch_size=None,
        language=None,
        task=None,
        return_timestamps=None,
        return_language=None,
        max_length=None,
    ):
        effective_batch_size = batch_size if batch_size is not None else self.batch_size
        if effective_batch_size % self.min_batch_size != 0:
            raise ValueError(f"Batch size must be a multiple of devices: {effective_batch_size} vs {self.min_batch_size}.")
        is_list_input = isinstance(inputs, list)
        if not is_list_input:
            inputs = [inputs]
        logger.info(f"Starting transcription for {len(inputs)} file(s)...")
        job_queue = queue.Queue(maxsize=effective_batch_size * 2)
        post_queue = queue.Queue() # Added post_queue
        futures_per_file = [[] for _ in inputs]
        final_results = [None] * len(inputs)
        
        batcher_thread = threading.Thread(
            target=self._batcher_worker,
            args=(job_queue, effective_batch_size, language, task, return_timestamps, max_length),
        )
        postprocessing_thread = threading.Thread( # Added postprocessing_thread
            target=self._postprocessing_worker, args=(post_queue, final_results)
        )
        batcher_thread.start()
        postprocessing_thread.start() # Started postprocessing_thread
        
        max_workers = common_config.get("preprocessing_workers") or min(32, (os.cpu_count() or 1) + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor: # Changed to single executor
            for i, file_path in enumerate(inputs):
                executor.submit( # Used single executor for preprocess
                    self._preprocess_worker,
                    file_path,
                    chunk_length_s,
                    stride_length_s,
                    job_queue,
                    futures_per_file[i],
                )
        
        job_queue.put(None)

        for i, file_futures in enumerate(futures_per_file):
            if not file_futures:
                logger.warning(f"File {i} ({inputs[i]}) produced no chunks.")
                post_queue.put((i, [], return_timestamps, return_language)) # Put to post_queue
                continue
            wait(file_futures)
            unpacked_outputs = []
            try:
                for future in file_futures:
                    pred_ids, stride_info = future.result()
                    unpacked_outputs.append(self._partial_postprocess(pred_ids, stride_info))
                post_queue.put((i, unpacked_outputs, return_timestamps, return_language)) # Put to post_queue
            except Exception as e:
                logger.error(f"Failed to retrieve results for file {i}: {e}")
                post_queue.put((i, e, return_timestamps, return_language)) # Put to post_queue
        
        post_queue.put(None) # Put None to post_queue
        batcher_thread.join()
        postprocessing_thread.join() # Joined postprocessing_thread
        
        logger.info("...transcription finished.")
        return final_results if is_list_input else final_results[0]

    def _execute_model(self, input_audio, attention_mask, language=None, task=None, return_timestamps=False, max_length=None):
        forced_decoder_ids = self.get_forced_decoder_ids(
            language=language, task=task, return_timestamps=return_timestamps
        )
        effective_max_length = max_length or self.max_length
        if self.is_sharded:
            output_ids = self.p_generate(self.params, input_audio, attention_mask, forced_decoder_ids, return_timestamps, effective_max_length)
        else:
            sharded_audio = shard(input_audio)
            sharded_mask = shard(attention_mask)
            output_ids = self.p_generate(self.params, sharded_audio, sharded_mask, forced_decoder_ids, return_timestamps, effective_max_length)
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
            
            return {"text": decoded_texts[0]}

    def _partial_postprocess(self, pred_ids, stride):
        out = {"tokens": pred_ids[None, :]}
        if stride:
            sr = self.feature_extractor.sampling_rate
            out["stride"] = (stride[0] / sr, stride[1] / sr, stride[2] / sr)
        return out

    def _preprocess_worker(self, file_path, chunk_length_s, stride_length_s, job_queue, futures_list):
        try:
            with open(file_path, "rb") as f:
                waveform = ffmpeg_read(f.read(), self.feature_extractor.sampling_rate)

            if waveform.shape[0] < self.feature_extractor.sampling_rate * 0.1: # Ignore very short files
                return

            chunk_len = round(chunk_length_s * self.feature_extractor.sampling_rate)

            # If the file is shorter than chunk_length_s, process it as a single chunk
            if waveform.shape[0] <= chunk_len:
                future = Future()
                futures_list.append(future)
                job = {"audio": waveform, "stride": (waveform.shape[0], 0, 0), "future": future}
                job_queue.put(job)
                return

            # For longer files, use the sliding window
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

            # Process the large batch in chunks of the max_bucket_size
            for i in range(0, len(batch), max_bucket_size):
                chunk = batch[i:i + max_bucket_size]
                actual_chunk_size = len(chunk)

                original_lengths = [job["audio"].shape[0] for job in chunk]

                # Pad the audio chunks to a fixed 30-second length
                max_audio_length = 30 * self.feature_extractor.sampling_rate # 480,000 samples
                padded_batch = np.zeros((len(chunk), max_audio_length), dtype=np.float32)
                for j, job in enumerate(chunk):
                    audio_chunk = job["audio"]
                    padded_batch[j, :audio_chunk.shape[0]] = audio_chunk

                # Create attention_mask from the original lengths
                feature_lengths = (np.array(original_lengths) // 160).astype(np.int32)
                # 3000 is the target_feature_length
                attention_mask = (np.arange(3000) < feature_lengths[:, None]).astype(np.int32)

                # Find the smallest bucket that can fit the current chunk
                padded_batch_size = next(
                    (b for b in self.BATCH_BUCKETS if b >= actual_chunk_size), 
                    max_bucket_size
                )
                pad_width = padded_batch_size - actual_chunk_size

                # Apply padding to padded_batch and attention_mask
                if pad_width > 0:
                    # pad padded_batch
                    feature_shape = padded_batch.shape[1:]
                    padding = np.zeros((pad_width, *feature_shape), dtype=padded_batch.dtype)
                    padded_batch = np.concatenate([padded_batch, padding], axis=0)
                    # pad attention_mask
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
        encoder_attention_implementation: str = "original", # Added
        decoder_attention_implementation: str = "original", # Added
    ):
        super().__init__(
            checkpoint,
            dtype,
            batch_size,
            max_length,
            encoder_attention_implementation=encoder_attention_implementation, # Passed
            decoder_attention_implementation=decoder_attention_implementation, # Passed
        )

    def shard_params(self, model_parallel_submesh=(2, 2, 1, 1)):
        """
        Switches the pipeline to `pjit` by generating the required `params_axes`
        metadata and then using the PjitPartitioner.
        """
        logger.info(f" Switching to `pjit` with model sharding. Mesh shape: {model_parallel_submesh}")

        logical_axis_rules = [
            ("batch", "data"),
            ("mlp", "model"),
            ("heads", "model"),
            ("vocab", None),
            ("embed", None),
            ("joined_kv", None),
            ("kv", None),
            ("length", None),
            ("num_mel", None),
            ("channels", None),
        ]

        partitioner = PjitPartitioner(
            model_parallel_submesh=model_parallel_submesh,
            logical_axis_rules=logical_axis_rules,
        )

        # Explicitly get the mesh and log it for visibility
        selected_mesh = partitioner.mesh
        logger.info(f"`default_mesh` selected device mesh with shape: {selected_mesh.shape} and axis_names: {selected_mesh.axis_names}")
        logger.info(f"Device mesh grid:\n{selected_mesh.devices}")

        logger.info("Generating `params_axes` metadata using `jax.eval_shape`...")
        
        def init_fn():
            input_shape = (1, self.config.num_mel_bins, self.config.max_source_positions * 2)
            input_features = jnp.zeros(input_shape, dtype="f4")
            decoder_input_ids = jnp.zeros((1, 1), dtype="i4")
            decoder_position_ids = jnp.broadcast_to(jnp.arange(decoder_input_ids.shape[-1]), decoder_input_ids.shape)
            
            rng = jax.random.PRNGKey(0)
            params_rng, dropout_rng = jax.random.split(rng)
            rngs = {"params": params_rng, "dropout": dropout_rng}

            return self.model.module.init(
                rngs,
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=jnp.ones_like(decoder_input_ids),
                decoder_position_ids=decoder_position_ids,
            )

        variables = jax.eval_shape(init_fn)
        param_axes = variables["params_axes"]
        logger.info("`params_axes` metadata generated successfully.")

        state = InferenceState(
            step=jnp.array(0),
            params=self.params, # Use self.params directly
            params_axes=freeze(param_axes)
        )
        
        mesh_axes = partitioner.get_mesh_axes(state)
        params_spec = mesh_axes.params

        def generate_fn(params, input_audio, forced_decoder_ids, return_timestamps, max_length):
            # on-device feature extraction
            input_features = jax.vmap(self._jax_feature_extractor, in_axes=(0, None, None, None, None, None, None, None, None))(
                input_audio,
                self.feature_extractor.sampling_rate,
                self.feature_extractor.n_fft,
                self.feature_extractor.hop_length,
                self.feature_extractor.n_fft,
                self.feature_extractor.feature_size,
                0.0,
                8000.0,
                3000 # target_feature_length
            )
            output_ids = self.model.generate(
                input_features,
                params=params,
                forced_decoder_ids=forced_decoder_ids,
                return_timestamps=return_timestamps,
                max_length=max_length,
            ).sequences
            return output_ids

        self.p_generate = partitioner.partition(
            generate_fn,
            in_axis_resources=(params_spec, P("data")),
            out_axis_resources=P("data"),
            static_argnums=(2, 3, 4),
        )

        logger.info("Re-sharding parameters for model parallelism...")
        unreplicated_params = jax_utils.unreplicate(self.params)
        self.params = partitioner.partition(
            lambda x: x, (params_spec,), params_spec
        )(freeze(unreplicated_params))
        
        
        
        jax.clear_caches()

        self.is_sharded = True
        logger.info("Model successfully sharded. Pipeline is now in `pjit` mode.")

def create_pipeline(mode="pmap", **kwargs):
    if mode == "pmap":
        return FlaxWhisperPmapPipeline(**kwargs)
    elif mode == "pjit":
        return FlaxWhisperPjitPipeline(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")