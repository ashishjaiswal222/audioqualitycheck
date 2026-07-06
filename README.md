# Voice/Audio Verification API

A production-grade, privacy-first Python microservice designed as a strict quality gatekeeper for downstream AI voice cloning, dubbing, and text-to-speech (TTS) pipelines. 

This API runs entirely offline, processing audio in-memory against 9 unique biometric, spectral, and machine learning heuristics (including Faster-Whisper, SpeechBrain, and PANNs) to ensure only pristine, studio-quality data enters your training sets.

## Core Capabilities
- **Background Noise & Music Detection:** Rejects files with low SNR or musical backing tracks.
- **Overlapping Speech (Crosstalk):** Uses spectral heuristics to detect if two people are speaking over each other.
- **Speaker Diarization:** Uses ECAPA-TDNN biometric embeddings to reject single files containing multiple speakers, and ensures identity consistency across batches.
- **Clarity & Intelligibility:** Leverages Whisper to calculate log-probabilities of speech, rejecting mumbled or distorted tracks.
- **Loudness & Pacing:** Enforces LUFS broadcast standards and calculates Words Per Minute (WPM).
- **Duration Constraints:** Hard limit of 30 seconds max to prevent expensive CPU inference loops on oversized files.

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+** (Tested natively on Windows/Linux)
- **uv**: The ultra-fast Python package manager (`pip install uv`)
- **FFmpeg**: Required in your system PATH for audio decoding.

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ashishjaiswal222/audioqualitycheck.git
   cd audioqualitycheck
   ```

2. **Install Dependencies via `uv`**
   ```bash
   uv venv
   # Activate your virtual environment:
   # On Windows: .venv\Scripts\activate
   # On Linux: source .venv/bin/activate
   uv pip install -r requirements.txt
   ```
   *(Note: The `requirements.txt` will pull all required PyTorch and audio analysis libraries automatically).*

---

## 💻 Running the Server

### Operating System Prerequisites (Linux / Docker)
If you are deploying this API to a Linux server (e.g. Ubuntu, Debian) or a Docker container, you must ensure that the C-library `libsndfile1` is installed so `soundfile` can decode `.wav` and `.flac` files natively.

```bash
sudo apt-get update && sudo apt-get install -y libsndfile1
```

*(Note: Windows users do not need to do this, as the library ships natively with the python wheel.)*

### Starting the Application
To start the API locally, you can use the provided `start.bat` file (on Windows), or run it manually via Uvicorn:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> **FFmpeg Zero-Touch Setup**: PyDub and Librosa require `ffmpeg` to decode `.mp3` and `.m4a` files. This repository features a completely automated Zero-Touch setup! When you run `start.bat` OR manually start `uvicorn`, the application will seamlessly check if FFmpeg is installed. If it is missing, the system will instantly download the correct pre-compiled binaries for Windows or Linux and place them into `scripts/bin/`. You do not need to install anything globally on your production servers!

Once the "Warming up models..." log finishes, the server is ready! 
Visit the interactive Swagger UI documentation at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧪 Testing Guide (API Interaction)

You can easily test the API using `curl` or Postman. 

### 1. Health Check
Verify the server and models are fully loaded.
```bash
curl -X GET "http://localhost:8000/api/v1/audio-verify/health"
```

### 2. Single Audio Check
Upload an audio file to evaluate its quality against all 9 metrics.
```bash
curl -X POST "http://localhost:8000/api/v1/audio-verify/check" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_audio_file.wav"
```

### 3. Batch Verification
Upload multiple takes simultaneously. The API will process each individually, and then compare their biometric voice embeddings to ensure the exact same person spoke across all files.
```bash
curl -X POST "http://localhost:8000/api/v1/audio-verify/check-batch" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@take1.wav" \
  -F "files=@take2.wav"
```

## 📚 Documentation
For a deep dive into the architecture, the specific models used, and the pipeline workflow, please see the markdown files located in the `docs/` folder.
