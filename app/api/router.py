from fastapi import APIRouter
from app.api.routes_audio_verify import router as audio_router

api_router = APIRouter()
api_router.include_router(audio_router, prefix="/audio-verify", tags=["verification"])
