# Architecture Overview

The Voice/Audio Verification API is a production-grade, offline microservice designed to validate and verify audio recordings before they are ingested into downstream AI voice cloning or dubbing pipelines.

## Core Design Principles

1. **100% Offline and Privacy-First**: 
   The system relies entirely on locally downloaded, open-source neural network weights (Whisper, ECAPA-TDNN, PANNs, Silero VAD). No audio data is ever sent to third-party cloud APIs (like OpenAI or AWS), ensuring complete data privacy for user biometrics.
   
2. **Fileless In-Memory Processing**: 
   Audio files are uploaded via standard `multipart/form-data`. They are instantly decoded into NumPy float arrays using `librosa`/`soundfile` and processed purely in memory. Temporary files are strictly avoided unless a specific library strictly demands a file path (e.g., Pyannote fallback).

3. **High-Concurrency Non-Blocking Design**: 
   Neural network inference (especially Whisper and ECAPA-TDNN) is fundamentally CPU-bound and highly blocking. To prevent a single heavy request from freezing the entire FastAPI event loop (which would cause other clients to hang or timeout), the system employs a `ProcessPoolExecutor`. Heavy verification tasks are shipped to background worker processes.

## Component Layout

- **`app/api/`**: Contains the FastAPI router and endpoint definitions (`/check` and `/check-batch`).
- **`app/schemas/`**: Pydantic models enforcing strict input validation and shaping the JSON output responses.
- **`app/utils/`**: Helper scripts for audio I/O, error handling, and file parsing.
- **`app/verification/`**: The core AI analysis logic. Each verification module (e.g., `vad_silence.py`, `transcription_asr.py`) encapsulates a specific domain check and its associated neural model.
- **`app/workers/`**: Manages the `ProcessPoolExecutor` lifecycle, starting up with the application server and shutting down gracefully.
- **`tests/`**: A comprehensive `pytest` suite simulating real-world dirty audio data using programmatic fixtures.

## Initialization Lifecycle

1. **Boot**: FastAPI boots and triggers the `@asynccontextmanager` lifespan.
2. **Warmup**: `models_loader.py` is invoked. It eagerly loads Whisper, Silero VAD, SpeechBrain ECAPA-TDNN, and PANNs into memory. This prevents the "cold-start" penalty on the first user request.
3. **Worker Pool**: The pre-warmed memory state is inherited by the `ProcessPoolExecutor` workers.
4. **Ready**: The `/health` endpoint flips to `{"status": "ok"}`, signaling load balancers that the node is ready for traffic.
