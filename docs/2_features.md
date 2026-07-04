# System Features

The Voice/Audio Verification API acts as a gatekeeper, analyzing raw audio to determine if it meets the stringent quality requirements for downstream AI voice cloning.

## 1. Single-Speaker Enforcement (Diarization)
The system leverages SpeechBrain's ECAPA-TDNN to map audio segments into a high-dimensional vector space, clustering distinct voices. If multiple significant clusters are found, the audio is flagged as containing multiple speakers, rejecting it to prevent cross-contamination in the cloning pipeline.

## 2. Crosstalk / Overlap Detection
Using either the Hugging Face `pyannote` pipeline or a robust spectral-peak fallback heuristic, the system detects frames where two people are speaking simultaneously. Crosstalk makes audio completely unusable for AI training.

## 3. Speech Clarity & Intelligibility
By utilizing Faster-Whisper, the API actually attempts to transcribe the audio. It measures the `avg_logprob` (the neural network's confidence in its own transcription). If the confidence drops below `-0.8`, the audio is flagged as mumbled, distorted, or completely unintelligible.

## 4. Background Music & Noise
- **Music**: The PANNs (Pre-trained Audio Neural Networks) CNN14 model analyzes the spectral features specifically looking for musical signatures (AudioSet class 137). If music is playing underneath the speech, it is rejected.
- **Noise**: A traditional Signal-to-Noise Ratio (SNR) is computed by comparing the RMS energy of speech regions (identified by Silero VAD) against the energy of non-speech regions.

## 5. Loudness & Clipping
- **LUFS**: The API ensures the overall integrated loudness of the track meets standard broadcast levels (minimum -35 LUFS).
- **Clipping**: Audio that peaks and distorts (ratio > 0.01 at absolute 0.99 amplitude) is rejected to prevent ruined training data.

## 6. Pace and Tone Consistency (Advisory)
- **Pace**: Calculates the words-per-minute (WPM). Rejects audio that is unnaturally slow (< 100 WPM) or alerts on audio that is excessively fast (> 200 WPM).
- **Tone**: Uses pitch contour mapping to flag highly monotone deliveries or unnatural robotic splices (pitch jumps > 50Hz).

## 7. Cross-File Batch Verification
The `/check-batch` endpoint takes multiple audio files (e.g., Take 1, Take 2, Take 3) and compares the speaker embeddings across all files using Cosine Distance. This guarantees that a user didn't accidentally upload their friend's voice for "Take 3".
