from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class Reason(BaseModel):
    code: str
    message: str

class VerificationResult(BaseModel):
    passed: bool
    confidence: float
    duration_seconds: float
    languages_detected: List[str]
    primary_reason: Optional[Reason] = None
    checks: Dict[str, Any]
    processed_in_ms: int
    
class BatchVerificationResult(BaseModel):
    session_id: Optional[str] = None
    overall_passed: bool
    results: Dict[str, VerificationResult]
    speaker_consistency: Dict[str, Any]
    processed_in_ms: int
