import time
import io
import asyncio
import logging
import os
import warnings
import numpy as np
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, status
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
        pipeline = FlaxWhisperOnlinePipeline("openai/whisper-large-v3", dtype=jnp.bfloat16)
        
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

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Query(None, description="Language of the audio."),
    task: str = Query("transcribe", enum=["transcribe", "translate"]),
    return_timestamps: bool = Query(False, description="Return timestamps."),
):
    logger.info(f"📥 Received request. Task: {task}, Language: {language}")
    if not is_pipeline_ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Pipeline is not ready.")

    try:
        # Read file content as bytes
        file_contents = await file.read()
        
        start_time = time.time()
        
        # Submit bytes directly to the pipeline
        # The pipeline handles decoding asynchronously
        future = pipeline(
            file_contents,
            language=language,
            task=task,
            return_timestamps=return_timestamps,
        )
        
        # Await result non-blocking
        result = await asyncio.wrap_future(future)
        
        end_time = time.time()
        
        # Calculate metrics
        pipeline_time = end_time - start_time
        total_samples = result.get("total_samples", 0)
        audio_duration_s = total_samples / 16000.0
        rtfx = audio_duration_s / pipeline_time if pipeline_time > 0 else float("inf")

        return {
            "transcription": result, # Contains "text" and "total_samples"
            "performance": {
                "pipeline_execution_time_s": round(pipeline_time, 4),
                "audio_duration_s": round(audio_duration_s, 4),
                "rtfx": round(rtfx, 4)
            }
        }
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
