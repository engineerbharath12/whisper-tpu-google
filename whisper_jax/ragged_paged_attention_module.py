import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from whisper_jax import layers
from .ragged_paged_adapter import ragged_paged_attention_adapter
from .splash_adapter import splash_attention_adapter

class FlaxWhisperRaggedPagedAttention(nn.Module):
    config: dict
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    causal: bool = False
    bias: bool = True
    dtype: jnp.dtype = jnp.float32
    params_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.head_dim = self.embed_dim // self.num_heads
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

    def __call__(
        self,
        hidden_states,
        key_value_states=None,
        attention_mask=None,
        init_cache=False,
        deterministic=True,
        # Paged Attention extras passed via kwargs or piggybacked
        cache_view=None, 
        cache_metadata=None
    ):
        """
        Hybrid Attention:
        - Cross-Attention (key_value_states != None): Use Splash/Standard.
        - Self-Attention (key_value_states == None): Use Ragged Paged.
        """
        is_cross = key_value_states is not None
        
        # --- 1. Cross-Attention (Fallback to Splash/Standard) ---
        if is_cross:
            # Standard projection logic
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)
            
            # Reshape
            query_states = query_states.reshape(query_states.shape[:2] + (self.num_heads, self.head_dim))
            key_states = key_states.reshape(key_states.shape[:2] + (self.num_heads, self.head_dim))
            value_states = value_states.reshape(value_states.shape[:2] + (self.num_heads, self.head_dim))
            
            # Use Splash Adapter for Cross-Attn
            attn_output = splash_attention_adapter(
                query_states, key_states, value_states, attention_mask, causal=False
            )
            return self.out_proj(attn_output.reshape(attn_output.shape[:2] + (self.embed_dim,))), None

        # --- 2. Self-Attention (Ragged Paged) ---
        # Requirement: cache_view and cache_metadata MUST be provided.
        # In the pipeline, we will pass these.
        
        if cache_view is None or cache_metadata is None:
            # This should not happen in the "Forced" pipeline.
            raise ValueError("Ragged Paged Attention requires `cache_view` and `cache_metadata`.")

        # Project Query only (K/V are retrieved from cache or appended to it)
        # In this simplified view, we assume the `ragged_page_attention` kernel 
        # handles reading.
        # But wait, we need to WRITE the current token's K/V to the cache first!
        
        # Projections
        query_states = self.q_proj(hidden_states) # [B, 1, E]
        key_states = self.k_proj(hidden_states)   # [B, 1, E]
        value_states = self.v_proj(hidden_states) # [B, 1, E]
        
        # Reshape
        query_states = query_states.reshape(query_states.shape[:2] + (self.num_heads, self.head_dim))
        key_states = key_states.reshape(key_states.shape[:2] + (self.num_heads, self.head_dim))
        value_states = value_states.reshape(value_states.shape[:2] + (self.num_heads, self.head_dim))

        # Update Cache (Write Step)
        # EasyDeL's kernel is read-only (Attention). We need a separate "Update Page" kernel
        # or use JAX scatter.
        # For now, we will assume `kv_pages` is mutable-ish or updated via a scatter.
        # kv_pages = cache_view.update(key_states, value_states, cache_metadata)
        
        # Call Adapter
        attn_output = ragged_paged_attention_adapter(
            query=query_states,
            kv_pages=cache_view.kv_pages, # Assumes updated
            pages_tables=cache_metadata.pages_tables,
            context_lens=cache_metadata.context_lens,
            num_seqs=cache_metadata.num_seqs,
            head_dim=self.head_dim
        )
        
        # Out Project
        attn_output = self.out_proj(attn_output.reshape(attn_output.shape[:2] + (self.embed_dim,)))
        return attn_output, None
