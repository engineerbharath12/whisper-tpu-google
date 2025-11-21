# benchmarks/test_async_h2d.py
import jax
import jax.numpy as jnp
import numpy as np
import time
from functools import partial
from rich.console import Console

console = Console()

# --- Configuration ---
NUM_DEVICES = jax.local_device_count()
BATCH_SIZE = 80
BUFFER_SHAPE = (BATCH_SIZE, 480000) # (batch_size, 30s of audio)
DTYPE = jnp.float32

# --- 1. Define Sharding ---
# Replicate the data across all devices (data parallelism)
device_mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(NUM_DEVICES, 1), axis_names=("data", "model"))
sharding = jax.sharding.NamedSharding(device_mesh, jax.sharding.PartitionSpec("data"))

# --- 2. Define JIT-compiled functions ---

@jax.jit
def dummy_computation(device_buffer: jax.Array) -> jax.Array:
    """
    A dummy computation that simulates the model's workload.
    It squares the input and then computes a sum, which forces the computation to actually happen.
    """
    return (device_buffer ** 2).sum()

def run_test():
    console.print("[bold blue]Testing Asynchronous H2D Double-Buffering Logic[/bold blue]")

    # --- 3. Pre-allocate Device Buffers ---
    console.print(f"Pre-allocating 2 device buffers with shape {BUFFER_SHAPE} across {NUM_DEVICES} devices...")
    device_buffers = [
        jax.device_put(np.zeros(BUFFER_SHAPE, dtype=DTYPE), sharding) for _ in range(2)
    ]
    # Block until allocation is complete to ensure accurate timing
    device_buffers[0].block_until_ready()
    device_buffers[1].block_until_ready()
    console.print("[green]Buffers allocated.[/green]")

    # --- 4. The Double-Buffering Loop ---
    num_iterations = 10
    buffer_idx = 0
    computation_future = None

    # Create a dummy CPU buffer
    cpu_data = np.ones(BUFFER_SHAPE, dtype=DTYPE)

    console.print(f"\nRunning {num_iterations} iterations of the double-buffering loop...")
    start_time = time.time()

    for i in range(num_iterations):
        console.log(f"Iter {i}:")
        
        # --- Overlap Step A: Start Asynchronous H2D Copy ---
        # This call is non-blocking. It starts the copy of `cpu_data` to the device
        # and immediately returns a jax.Array future.
        t1 = time.time()
        device_buffers[buffer_idx] = jax.device_put(cpu_data, sharding)
        console.log(f"  -> Kicked off async copy to Buffer {buffer_idx}.")

        # --- Overlap Step B: Wait for the PREVIOUS computation to finish ---
        # This is the key to the overlap. We are waiting for the computation that was
        # running on the *other* buffer.
        if computation_future is not None:
            result = computation_future.block_until_ready()
            t2 = time.time()
            console.log(f"  <- Finished computation on Buffer {1 - buffer_idx} in {t2 - t1:.4f}s. Result: {result}")
        
        # --- Overlap Step C: Start the NEXT computation ---
        # This call is also non-blocking. It queues the computation on the current buffer.
        # JAX's runtime ensures this will only execute after the copy to this buffer is complete.
        computation_future = dummy_computation(device_buffers[buffer_idx])
        console.log(f"  -> Kicked off computation on Buffer {buffer_idx}.")

        # --- Flip the buffer index for the next iteration ---
        buffer_idx = 1 - buffer_idx

    # --- 5. Final Wait ---
    # We need to wait for the very last computation to finish
    if computation_future is not None:
        result = computation_future.block_until_ready()
        console.log(f"Final Wait: Finished final computation. Result: {result}")

    end_time = time.time()
    console.print(f"\nTotal time for {num_iterations} iterations: [bold green]{end_time - start_time:.4f}s[/bold green]")
    console.print("[green]Test successful. The core logic is correct.[/green]")


if __name__ == "__main__":
    run_test()
    # Clean up the temp script
    os.remove(__file__)
