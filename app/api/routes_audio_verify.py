import time
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from app.workers.pool import run_in_pool
from app.verification.pipeline import run_audio_verification
from app.utils.errors import AudioVerificationError
from app.schemas.responses import VerificationResult, BatchVerificationResult
from app.verification.constants import ReasonCode, DIARIZATION_DISTANCE_THRESHOLD
import structlog
import asyncio
import numpy as np

def sanitize_numpy(obj):
    if isinstance(obj, dict):
        return {k: sanitize_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_numpy(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

logger = structlog.get_logger()
router = APIRouter()

@router.post("/check", response_model=VerificationResult)
async def check_audio(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    strict_tone_check: bool = Form(False)
):
    start_time = time.time()
    try:
        file_bytes = await file.read()
        filename = file.filename
        
        from app.config import settings
        result_dict = await asyncio.wait_for(
            run_in_pool(
                run_audio_verification,
                file_bytes=file_bytes,
                filename=filename,
                strict_tone=strict_tone_check
            ),
            timeout=settings.request_timeout_seconds
        )
        
        if "embedding" in result_dict:
            del result_dict["embedding"]
            
        processed_in_ms = int((time.time() - start_time) * 1000)
        result_dict["processed_in_ms"] = processed_in_ms
        
        return JSONResponse(status_code=200, content=sanitize_numpy(result_dict))
        
    except asyncio.TimeoutError:
        logger.warning("Verification Timeout")
        from app.config import settings
        raise HTTPException(status_code=408, detail={"code": "REQUEST_TIMEOUT", "message": f"Verification timed out after {settings.request_timeout_seconds} seconds."})
    except AudioVerificationError as e:
        logger.warning("AudioVerificationError", code=e.code, message=e.message)
        raise HTTPException(status_code=e.status_code, detail={"code": e.code, "message": e.message})
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error", "traceback": err_msg})

@router.post("/check-batch", response_model=BatchVerificationResult)
async def check_batch_audio(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    strict_tone_check: bool = Form(False)
):
    start_time = time.time()
    
    tasks = []
    file_names = []
    
    for f in files:
        file_bytes = await f.read()
        file_names.append(f.filename)
        tasks.append(
            run_in_pool(
                run_audio_verification,
                file_bytes=file_bytes,
                filename=f.filename,
                strict_tone=strict_tone_check
            )
        )
        
    try:
        from app.config import settings
        results_list = await asyncio.wait_for(
            asyncio.gather(*tasks),
            timeout=settings.request_timeout_seconds
        )
    except asyncio.TimeoutError:
        from app.config import settings
        raise HTTPException(status_code=408, detail={"code": "REQUEST_TIMEOUT", "message": f"Verification timed out after {settings.request_timeout_seconds} seconds."})
    except AudioVerificationError as e:
        raise HTTPException(status_code=e.status_code, detail={"code": e.code, "message": e.message})
        
    results = {}
    embeddings = {}
    
    overall_passed = True
    for i, res in enumerate(results_list):
        fname = file_names[i]
        if not res["passed"]:
            overall_passed = False
        
        if res.get("embedding") is not None:
            embeddings[fname] = res["embedding"]
            
        res_copy = res.copy()
        if "embedding" in res_copy:
            del res_copy["embedding"]
        results[fname] = res_copy
        
    speaker_consistency = {"passed": True}
    
    if overall_passed and len(embeddings) > 1:
        from scipy.spatial.distance import cosine
        pairwise = {}
        files_with_emb = list(embeddings.keys())
        for i in range(len(files_with_emb)):
            for j in range(i+1, len(files_with_emb)):
                f1 = files_with_emb[i]
                f2 = files_with_emb[j]
                
                dist = cosine(embeddings[f1], embeddings[f2])
                pairwise[f"{f1}_{f2}"] = round(float(dist), 3)
                
                if dist > DIARIZATION_DISTANCE_THRESHOLD:
                    speaker_consistency["passed"] = False
                    speaker_consistency["reason_code"] = ReasonCode.SPEAKER_MISMATCH_ACROSS_FILES.value
                    
        speaker_consistency["pairwise_distance"] = pairwise
        if not speaker_consistency["passed"]:
            overall_passed = False

    processed_in_ms = int((time.time() - start_time) * 1000)
    
    return JSONResponse(status_code=200, content=sanitize_numpy({
        "session_id": session_id,
        "overall_passed": overall_passed,
        "results": results,
        "speaker_consistency": speaker_consistency,
        "processed_in_ms": processed_in_ms
    }))
