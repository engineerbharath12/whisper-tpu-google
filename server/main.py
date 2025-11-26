import time
import io
import asyncio
import logging
import os
import warnings
import numpy as np
import uuid
from typing import Optional, AsyncGenerator
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, status, Request
import jax
import jax.numpy as jnp
from whisper_jax.pipeline_online import FlaxWhisperOnlinePipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Whisper JAX FastAPI Server",
    description="A high-performance API for audio transcription using the Whisper JAX pipeline.",
)

# Global placeholder for the pipeline and readiness flag
pipeline: Optional[FlaxWhisperOnlinePipeline] = None
is_pipeline_ready = False

@app.on_event("startup")
async def startup_event():
    """
    On server startup, initialize the model pipeline and set the readiness flag.
    """
    global pipeline, is_pipeline_ready
    
    # --- JAX Cache Configuration ---
    try:
        JAX_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".jax_cache")
        os.makedirs(JAX_CACHE_DIR, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)
        logger.info(f"JAX Cache enabled. Using directory: {JAX_CACHE_DIR}")
    except ImportError:
        warnings.warn("Could not configure JAX cache.")
        
    logger.info("🚀 Initializing FlaxWhisperOnlinePipeline...")
    try:
        # Initialize the optimized online pipeline
        # The pipeline performs its own comprehensive warm-up on init
        pipeline = FlaxWhisperOnlinePipeline("openai/whisper-large-v3-turbo", dtype=jnp.bfloat16)
        
        is_pipeline_ready = True
        logger.info("✅ Pipeline initialized and warmed up successfully.")
    except Exception as e:
        logger.error(f"💥 Pipeline initialization or warm-up failed: {e}", exc_info=True)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/ready")
async def ready_check():
    if not is_pipeline_ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Pipeline is not ready.")
    return {"status": "ready"}

# --- New: Streaming Transcription Endpoint ---
@app.post("/transcribe_stream")
async def transcribe_stream(
    request: Request,
    language: Optional[str] = Query(None, description="Language of the audio."),
    task: str = Query("transcribe", enum=["transcribe", "translate"]),
    return_timestamps: bool = Query(False, description="Return timestamps."),
):
    logger.info(f"📥 Received streaming request. Task: {task}, Language: {language}")
    if not is_pipeline_ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Pipeline is not ready.")

    try:
        request_id = str(uuid.uuid4()) # Generate request_id here
        future = asyncio.Future() # Create a future to await the result
        pipeline.requests[request_id] = {
            "future": future,
            "chunks": [],
            "chunk_count": 0,
            "kwargs": {"language": language, "task": task, "return_timestamps": return_timestamps},
            "total_samples": 0
        }

        # Submit the streaming preprocessing task
        asyncio.create_task(
            pipeline.process_streaming_request(
                request_id, 
                {"language": language, "task": task, "return_timestamps": return_timestamps}, 
                request.stream() # Pass the async iterator
            )
        )
        
        start_time = time.time()
        result = await future # Await the completion of this specific request
        end_time = time.time()

        pipeline_time = end_time - start_time
        audio_duration_s = result.get("total_samples", 0) / 16000.0 # Total samples processed by ffmpeg (sped up)
        rtfx = audio_duration_s / pipeline_time if pipeline_time > 0 else float("inf")

        return {
            "transcription": result,
            "performance": {
                "pipeline_execution_time_s": round(pipeline_time, 4),
                "audio_duration_s": round(audio_duration_s, 4),
                "rtfx": round(rtfx, 4)
            }
        }
    except Exception as e:
        logger.error(f"Streaming transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# --- Existing: One-Shot File Upload Endpoint ---
@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Query(None, description="Language of the audio."),
    task: str = Query("transcribe", enum=["transcribe", "translate"]),
    return_timestamps: bool = Query(False, description="Return timestamps."),
):
    logger.info(f"📥 Received buffered request. Task: {task}, Language: {language}")
    if not is_pipeline_ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Pipeline is not ready.")

    try:
        # Read file content as bytes (buffers entire file)
        file_contents = await file.read()
        
        start_time = time.time()
        result_future = pipeline(file_contents, language=language, task=task, return_timestamps=return_timestamps)
        result = await asyncio.wrap_future(result_future)
        end_time = time.time()
        
        pipeline_time = end_time - start_time
        total_samples = result.get("total_samples", 0)
        audio_duration_s = total_samples / 16000.0
        rtfx = audio_duration_s / pipeline_time if pipeline_time > 0 else float("inf")

        return {
            "transcription": result,
            "performance": {
                "pipeline_execution_time_s": round(pipeline_time, 4),
                "audio_duration_s": round(audio_duration_s, 4),
                "rtfx": round(rtfx, 4)
            }
        }
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")