# API Reference

## 1. Health Check

Checks if the server is up and if the machine learning models have finished loading into the worker pools.

- **URL**: `/api/v1/audio-verify/health`
- **Method**: `GET`
- **Response `200 OK`**:
  ```json
  {
    "status": "ok",
    "models_loaded": true
  }
  ```
- **Response `503 Service Unavailable`**:
  ```json
  {
    "status": "starting_up",
    "models_loaded": false
  }
  ```

---

## 2. Single Audio Check

Analyzes a single audio file against all 9 verification heuristics.

- **URL**: `/api/v1/audio-verify/check`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`

### Form Data Parameters
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | Yes | The audio file (WAV, MP3, M4A, etc.) |
| `session_id` | String | No | Optional identifier for tracking |
| `strict_tone_check` | Boolean | No | If true, fails the audio on unnatural pitch instead of just advising. |

### Example Response (`200 OK`)
```json
{
  "passed": false,
  "confidence": 0.0,
  "duration_seconds": 15.2,
  "languages_detected": ["en"],
  "primary_reason": {
    "code": "BACKGROUND_MUSIC_DETECTED",
    "message": "Background music detected. Please provide a voice-only recording with no music."
  },
  "checks": {
    "silence_presence": { "passed": true, "score": 0.85 },
    "noise": { "passed": true, "score": 45.2, "unit": "dB_SNR" },
    "loudness_clipping": { "passed": true, "lufs": -22.1, "clipping_ratio": 0.0 },
    "background_music": { "passed": false, "score": 0.72 },
    "overlapping_voices": { "passed": true, "overlap_seconds": 0.0 },
    "multiple_speakers": { "passed": true, "num_clusters": 1 },
    "speech_clarity": { "passed": true, "avg_confidence": -0.2 },
    "speech_pace": { "passed": true, "words_per_minute": 135 },
    "tone_consistency": { "passed": true, "advisory": true }
  },
  "processed_in_ms": 1250
}
```

---

## 3. Batch Verification (Speaker Consistency)

Analyzes multiple takes/files and ensures the exact same speaker was recorded across all files, in addition to running the standard heuristics on each file.

- **URL**: `/api/v1/audio-verify/check-batch`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`

### Form Data Parameters
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `files` | File Array | Yes | Multiple audio files to analyze and compare |
| `session_id` | String | No | Optional identifier |
| `strict_tone_check` | Boolean | No | Same as single check |

### Example Response (`200 OK`)
```json
{
  "session_id": null,
  "overall_passed": false,
  "speaker_consistency": {
    "passed": false,
    "reason_code": "SPEAKER_MISMATCH_ACROSS_FILES",
    "pairwise_distance": {
      "take1.wav_take2.wav": 0.842
    }
  },
  "results": {
    "take1.wav": {
       "passed": true,
       "checks": { ... }
    },
    "take2.wav": {
       "passed": true,
       "checks": { ... }
    }
  },
  "processed_in_ms": 3200
}
```
