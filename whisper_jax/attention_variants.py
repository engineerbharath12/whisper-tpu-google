# coding=utf-8
# Copyright 2023 The OpenAI Authors and The HuggingFace Inc. team. All rights reserved.
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
""" Flax whisper model."""

from functools import partial
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from jax import lax

from transformers import WhisperConfig

from whisper_jax import layers
from whisper_jax.layers import with_sharding_constraint, nd_dense_init
from flax.linen import partitioning as nn_partitioning
param_with_axes = nn_partitioning.param_with_axes
from .splash_adapter import splash_attention_adapter

class FlaxWhisperAttention(nn.Module):
    config: WhisperConfig
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    causal: bool = False
    bias: bool = True
    dtype: jnp.dtype = jnp.float32
    params_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        dense = partial(
            layers.DenseGeneral,
            self.embed_dim,
            axis=-1,
            dtype=self.dtype,
            params_dtype=self.params_dtype,
            kernel_axes=("embed", "joined_kv"),
        )

        self.q_proj = dense(use_bias=self.bias)
        self.k_proj = dense(use_bias=False)
        self.v_proj = dense(use_bias=self.bias)

        self.out_proj = layers.DenseGeneral(
            self.embed_dim,
            axis=-1,
            dtype=self.dtype,
            params_dtype=self.params_dtype,
            kernel_axes=("joined_kv", "embed"),
            use_bias=self.bias,
        )

        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_target_positions), dtype="bool"), dtype="bool"
            )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        key_value_states: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        is_cross_attention = key_value_states is not None
        batch_size = hidden_states.shape[0]

        query_states = self.q_proj(hidden_states)

        if is_cross_attention:
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)
        else:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)

        query_states = with_sharding_constraint(query_states, ("batch", "length", "heads", "kv"))
        key_states = with_sharding_constraint(key_states, ("batch", "length", "heads", "kv"))
        value_states = with_sharding_constraint(value_states, ("batch", "length", "heads", "kv"))

        if self.causal:
            query_length, key_length = query_states.shape[1], key_states.shape[1]
            if self.has_variable("cache", "cached_key"):
                mask_shift = self.variables["cache"]["cache_index"]
                # max_length of cached_key is last dim
                max_decoder_length = self.variables["cache"]["cached_key"].shape[-1]
                causal_mask = lax.dynamic_slice(
                    self.causal_mask,
                    (0, 0, mask_shift, 0),
                    (1, 1, query_length, max_decoder_length),
                )
            else:
                causal_mask = self.causal_mask[:, :, :query_length, :key_length]
            causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

        # combine masks if needed
        if attention_mask is not None and self.causal:
            attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
            attention_mask = combine_masks(attention_mask, causal_mask)
        elif self.causal:
            attention_mask = causal_mask
        elif attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.

        if self.causal and (self.has_variable("cache", "cached_key") or init_cache):
            key_states, value_states, attention_mask = self._concatenate_to_cache(
                key_states, value_states, query_states, attention_mask
            )

        # Convert the boolean attention mask to an attention bias.
        if attention_mask is not None:
            # attention mask in the form of attention bias
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
        else:
            attention_bias = None

        dropout_rng = None
        if not deterministic and self.dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

    def _split_heads(self, hidden_state) -> jnp.ndarray:
        return hidden_state.reshape(hidden_state.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_state) -> jnp.ndarray:
        return hidden_state.reshape(hidden_state.shape[:2] + (self.embed_dim,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        is_initialized = self.has_variable("cache", "cached_key")

        def swap_dims(x):
            return x[:-3] + tuple(x[i] for i in [-2, -1, -3])

        cached_key = self.variable("cache", "cached_key", jnp.zeros, swap_dims(key.shape), key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, swap_dims(value.shape), value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            batch_size, num_heads, head_dim, seq_length = cached_key.value.shape
            num_updated_cache_vectors = query.shape[1]
            expected_shape = (batch_size, 1, num_heads, head_dim)
            if num_updated_cache_vectors == 1 and expected_shape != query.shape:
                raise ValueError(
                    f"Autoregressive cache shape error, expected query shape {expected_shape} instead got {query.shape}"
                )

            cur_index = cache_index.value
            one_token_key = jnp.moveaxis(key, -3, -1)
            one_token_value = jnp.moveaxis(value, -3, -1)

            if num_updated_cache_vectors > 1:
                indices = jnp.eye(num_updated_cache_vectors, seq_length)[None, None]
                key = cached_key.value + jnp.matmul(one_token_key, indices)
                value = cached_value.value + jnp.matmul(one_token_value, indices)
            else:
                one_hot_indices = jax.nn.one_hot(cur_index, seq_length, dtype=key.dtype)
                key = cached_key.value + one_token_key * one_hot_indices
                value = cached_value.value + one_token_value * one_hot_indices

            cached_key.value = key
            cached_value.value = value
            cache_index.value = cache_index.value + num_updated_cache_vectors

            key = jnp.moveaxis(key, -1, -3)
            value = jnp.moveaxis(value, -1, -3)

            pad_mask = jnp.broadcast_to(
                jnp.arange(seq_length) < cur_index + num_updated_cache_vectors,
                (batch_size,) + (1, num_updated_cache_vectors, seq_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)

        return key, value, attention_mask


class FlaxWhisperAttentionSplash(FlaxWhisperAttention):
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        key_value_states: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        is_cross_attention = key_value_states is not None
        batch_size, q_seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)

        if is_cross_attention:
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)
        else:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)

        query_states = with_sharding_constraint(query_states, ("batch", "length", "heads", "kv"))
        key_states = with_sharding_constraint(key_states, ("batch", "length", "heads", "kv"))
        value_states = with_sharding_constraint(value_states, ("batch", "length", "heads", "kv"))


        attn_output = splash_attention_adapter(
                query_states, 
                key_states, 
                value_states, 
                attention_mask=attention_mask, 
                causal=self.causal
            )
        attn_weights = None  # Splash attention does not return weights
   
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

