from fastapi import FastAPI
from fastapi.responses import JSONResponse
import inspect
original_getframeinfo = inspect.getframeinfo
def patched_getframeinfo(frame, context=1):
    info = original_getframeinfo(frame, context)
    if hasattr(info, 'filename') and info.filename.endswith("\\inspect.py"):
        return inspect.Traceback(info.filename.replace("\\inspect.py", "/inspect.py"), info.lineno, info.function, info.code_context, info.index)
    return info
inspect.getframeinfo = patched_getframeinfo

import os
# Ensure ffmpeg downloaded in scripts/bin is available to all subprocesses
bin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "bin"))
if bin_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")

from contextlib import asynccontextmanager
from app.api.router import api_router
from app.workers.pool import start_worker_pool, stop_worker_pool, get_pool
import structlog

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up API...")
    
    # Auto-install FFmpeg if missing before starting the workers
    import subprocess
    import sys
    setup_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "setup_ffmpeg.py"))
    if os.path.exists(setup_script):
        logger.info("Checking FFmpeg installation...")
        subprocess.run([sys.executable, setup_script], check=True)
        
    start_worker_pool()
    yield
    logger.info("Shutting down API...")
    stop_worker_pool()

app = FastAPI(
    title="Voice/Audio Verification API",
    description="API for verifying voice recordings for cloning/dubbing pipelines.",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/api/v1/audio-verify/health")
async def health():
    try:
        pool = get_pool()
        if not pool:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return JSONResponse(status_code=200, content={"status": "ready"})
    except RuntimeError:
        return JSONResponse(status_code=503, content={"status": "not_ready"})
