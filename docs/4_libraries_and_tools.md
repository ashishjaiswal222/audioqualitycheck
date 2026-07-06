# Libraries and Tools

This project integrates several state-of-the-art open-source audio processing libraries into a single cohesive pipeline.

## Core Framework
- **FastAPI**: The asynchronous web framework driving the REST API. Chosen for its speed, automatic Swagger UI documentation, and native async support.
- **uv**: An ultra-fast Python package installer and resolver written in Rust. Used for managing dependencies and environments reliably across platforms.

## AI & Audio Processing Libraries

### 1. Silero VAD (Voice Activity Detection)
- **Source**: `torch.hub.load('snakers4/silero-vad', 'silero_vad')`
- **Purpose**: Fast, robust detection of spoken segments. We use it to cut out dead air and calculate the ratio of speech to silence.

### 2. Faster-Whisper
- **Source**: `faster-whisper` PyPI package (using CTranslate2 backend).
- **Purpose**: Extremely fast speech-to-text. We use the `"tiny"` model not for accurate subtitling, but to measure the neural network's `avg_logprob`—a mathematical proxy for "clarity" and "mumbling". If Whisper can't transcribe it confidently, it's not clear enough to clone.

### 3. SpeechBrain (ECAPA-TDNN)
- **Source**: `speechbrain.pretrained.EncoderClassifier`
- **Purpose**: Speaker Diarization and Verification. This library extracts a mathematical embedding ("voiceprint") from the audio. We cluster these embeddings within a single file to detect multiple speakers, and compare them across multiple files in the `/check-batch` endpoint to ensure identity consistency.

### 4. PANNs Inference (Pre-trained Audio Neural Networks)
- **Source**: `panns_inference` (CNN14)
- **Purpose**: Audio tagging. Trained on AudioSet, this model can detect 527 different classes of sounds. We use it strictly to detect class `137` (Music), rejecting files that have background music mixed in.

### 5. Librosa & Soundfile
- **Source**: `librosa`, `soundfile`
- **Purpose**: The backbone of audio I/O. Decodes MP3, WAV, M4A, resamples it dynamically, and provides mathematical features like STFT (Short-Time Fourier Transform) for overlap detection and `pyin` for pitch/tone extraction.

### 6. Pyloudnorm
- **Source**: `pyloudnorm`
- **Purpose**: Standardized broadcast loudness calculations (LUFS). Ensures the audio isn't completely quiet.

### 7. Pyannote.audio (Optional)
- **Source**: `pyannote/overlapped-speech-detection`
- **Purpose**: Heavyweight overlapped speech detection. The pipeline currently defaults to a spectral heuristic fallback to avoid Hugging Face auth token requirements, but Pyannote can be enabled via `USE_OVERLAP_HEURISTIC_FALLBACK=False`.

### 8. FFmpeg
- **Source**: `setup_ffmpeg.py` (downloads from `BtbN` or `johnvansickle`)
- **Purpose**: Required to decode compressed formats like `.mp3` and `.m4a` through `pydub`. It is automatically downloaded and installed natively within the API environment during the FastAPI lifespan boot phase.
