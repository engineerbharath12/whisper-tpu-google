import jax.numpy as jnp
import numpy as np
from typing import Optional

# Stub import for EasyDeL kernel
# In production, this must be installed: pip install easydel
try:
    from ejkernel.modules import ragged_page_attention
except ImportError:
    ragged_page_attention = None

def ragged_paged_attention_adapter(
    query: jnp.ndarray,
    kv_pages: jnp.ndarray,
    pages_tables: jnp.ndarray,
    context_lens: jnp.ndarray,
    num_seqs: int,
    head_dim: int,
    softmax_scale: float = None
):
    """
    Adapter for EasyDeL's Ragged Paged Attention Kernel.
    Forces the use of the Pallas kernel for TPU decoding.
    """
    if ragged_page_attention is None:
        raise ImportError("EasyDeL (ejkernel) not found. Please install it to use Ragged Paged Attention.")

    # Query: [Batch, 1, Heads, Dim]
    batch, q_len, num_heads, dim = query.shape
    
    # Flatten Query for Kernel: [Total_Tokens, Heads, Dim]
    query_ragged = query.reshape(-1, num_heads, dim)

    # Metadata
    # query_start_loc is trivial for decoding (1 token per seq): [0, 1, 2, ... N]
    query_start_loc = jnp.arange(batch + 1, dtype=jnp.int32)
    
    num_seqs_flat = jnp.array([num_seqs], dtype=jnp.int32).reshape(-1)

    if softmax_scale is None:
        softmax_scale = 1.0 / jnp.sqrt(dim)

    # Execute Kernel
    output_ragged = ragged_page_attention(
        query=query_ragged,
        kv_pages=kv_pages,
        context_lens=context_lens,
        pages_tables=pages_tables,
        query_start_loc=query_start_loc,
        num_seqs=num_seqs_flat,
        softmax_scale=softmax_scale,
        compute_dtype=jnp.bfloat16,
        optimized=True 
    )

    # Reshape Output: [Batch, 1, Heads, Dim]
    output = output_ragged.reshape(batch, q_len, num_heads, dim)
    return output
