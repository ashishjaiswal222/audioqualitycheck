import asyncio
from concurrent.futures import ProcessPoolExecutor
from app.config import settings
from app.verification.models_loader import init_worker_models
import structlog

logger = structlog.get_logger()
_pool: ProcessPoolExecutor | None = None

def start_worker_pool():
    global _pool
    logger.info("Starting ProcessPoolExecutor...", size=settings.worker_pool_size)
    _pool = ProcessPoolExecutor(
        max_workers=settings.worker_pool_size,
        initializer=init_worker_models
    )

def stop_worker_pool():
    global _pool
    if _pool:
        _pool.shutdown(wait=True)
        _pool = None

def get_pool() -> ProcessPoolExecutor:
    if not _pool:
        raise RuntimeError("Worker pool not initialized.")
    return _pool

async def run_in_pool(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    import functools
    pfunc = functools.partial(func, *args, **kwargs)
    pool = get_pool()
    return await loop.run_in_executor(pool, pfunc)
