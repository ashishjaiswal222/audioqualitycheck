@echo off
echo Starting Voice/Audio Verification API...
echo ------------------------------------------
echo ------------------------------------------

:: Ensure FFmpeg is installed and ready
echo Ensuring FFmpeg is installed...
uv run python scripts/setup_ffmpeg.py

:: Run the FastAPI server using uv
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

:: Pause so the window doesn't immediately close if the server crashes
pause
