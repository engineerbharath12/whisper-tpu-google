import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
from typing import Optional

from .splash_attention_new.splash_attention_kernel import make_splash_mha, BlockSizes, QKVLayout
from .splash_attention_new.splash_attention_mask import MultiHeadMask, FullMask, CausalMask, NumpyMask, LogicalAnd

@partial(jax.jit, static_argnames=("q_seq_len", "kv_seq_len", "multi_head_mask", "block_sizes"))
def _splash_attention_kernel(query, key, value, multi_head_mask, q_seq_len, kv_seq_len, block_sizes):
    """
    A JIT-compiled kernel function for splash attention.
    """
    head_dim = query.shape[-1]

    # --- 1. Scale Query ---
    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(query.dtype)

    # --- 2. Transpose for Kernel ---
    query_transposed = query.transpose(0, 2, 1, 3)
    key_transposed = key.transpose(0, 2, 1, 3)
    value_transposed = value.transpose(0, 2, 1, 3)

    # --- 3. Handle Sequence Padding ---
    BLOCK_SIZE = 128
    SEQ_AXIS = 2
    
    def pad_to_multiple(x, seq_len):
        pad_len = (BLOCK_SIZE - seq_len % BLOCK_SIZE) % BLOCK_SIZE
        if pad_len == 0:
            return x
        pad_width_seq = [(0, 0)] * x.ndim
        pad_width_seq[SEQ_AXIS] = (0, pad_len)
        return jnp.pad(x, pad_width_seq)

    q_padded_seq = pad_to_multiple(query_transposed, q_seq_len)
    k_padded_seq = pad_to_multiple(key_transposed, kv_seq_len)
    v_padded_seq = pad_to_multiple(value_transposed, kv_seq_len)

    # --- 4. Create and Execute the Kernel ---
    splash_mha_kernel = make_splash_mha(mask=multi_head_mask, head_shards=1, q_seq_shards=1, block_sizes=block_sizes)
    vmapped_kernel = jax.vmap(splash_mha_kernel, in_axes=(0, 0, 0), out_axes=0)
    
    output_padded = vmapped_kernel(q_padded_seq, k_padded_seq, v_padded_seq)

    # --- 5. Slice Padding and Reshape Output Back ---
    output_sliced_seq = output_padded[:, :, :q_seq_len, :]
    output_transposed = output_sliced_seq.transpose(0, 2, 1, 3)
    output = output_transposed[:, :, :, :head_dim]
    
    return output

def splash_attention_adapter(query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray, 
                             attention_mask: Optional[jnp.ndarray] = None, causal: bool = False):
    """
    A general splash attention adapter that can handle causal masking and attention masks.
    """
    batch_size, q_seq_len, num_heads, head_dim = query.shape
    _, kv_seq_len, _, _ = key.shape

    BLOCK_SIZE = 128
    padded_q_len = (q_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
    padded_kv_len = (kv_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
    mask_shape = (padded_q_len, padded_kv_len)

    if attention_mask is not None:
        # The JAX tracer should not be converted to a NumPy array
        attention_mask = jnp.array(attention_mask)
        if attention_mask.ndim == 2:
            attention_mask = jnp.expand_dims(attention_mask, axis=1)  # `[B, 1, S]`
        # Invert the mask for splash attention
        attention_mask = jnp.logical_not(attention_mask)
        
        padded_attention_mask = np.pad(attention_mask, 
                                     ((0, padded_q_len - q_seq_len), (0, padded_kv_len - kv_seq_len)), 
                                     mode='constant', constant_values=False)
        padding_mask = NumpyMask(padded_attention_mask)
        if causal:
            causal_mask = CausalMask(shape=mask_shape)
            mask = LogicalAnd(causal_mask, padding_mask)
        else:
            mask = padding_mask
    elif causal:
        mask = CausalMask(shape=mask_shape)
    else:
        mask = FullMask(mask_shape)

    multi_head_mask = MultiHeadMask([mask for _ in range(num_heads)])

    if q_seq_len <= 128:
        block_sizes = BlockSizes(
            block_q=128,
            block_kv=128,
            block_kv_compute=128,
            block_q_dkv=64,
            block_kv_dkv=64,
            block_kv_dkv_compute=64,
            q_layout=QKVLayout.SEQ_MINOR,
            k_layout=QKVLayout.SEQ_MINOR,
            v_layout=QKVLayout.SEQ_MINOR,
            use_fused_bwd_kernel=True,
        )
    else:
        block_sizes = BlockSizes(
            block_q=1536,
            block_kv=1536,
            block_kv_compute=1536,
            block_q_dkv=128,
            block_kv_dkv=1536,
            block_kv_dkv_compute=384,
            q_layout=QKVLayout.SEQ_MINOR,
            k_layout=QKVLayout.SEQ_MINOR,
            v_layout=QKVLayout.SEQ_MINOR,
            use_fused_bwd_kernel=True,
        )

    return _splash_attention_kernel(query, key, value, multi_head_mask, q_seq_len, kv_seq_len, block_sizes)