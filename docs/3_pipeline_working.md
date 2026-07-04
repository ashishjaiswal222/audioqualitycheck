# How the Pipeline Works

When an audio file hits the `/api/v1/audio-verify/check` endpoint, it undergoes a strict sequence of checks. The pipeline is designed to "fail fast", meaning if a fatal flaw (like total silence) is detected early, it immediately returns a failure without burning CPU cycles on complex models like Whisper or ECAPA-TDNN.

## Step-by-Step Execution

1. **Ingest & Normalize (`audio_io.py`)**
   - The file byte stream is caught in memory.
   - It is loaded via `librosa` or `soundfile`, instantly downmixed to Mono, and resampled to exactly `16,000 Hz`.
   - Re-sampling immediately guarantees all downstream AI models get the exact tensor dimensions they expect.

2. **Voice Activity Detection (`vad_silence.py`)**
   - The audio array is passed to **Silero VAD**.
   - Silero returns timestamp boundaries for every spoken word/phrase.
   - If the total spoken time is less than 30% of the track, the API rejects the file as `SILENCE_ONLY` or `INSUFFICIENT_SPEECH`.

3. **Noise Estimation (`noise_snr.py`)**
   - Using the boundaries from Silero, the audio is split into "speech" arrays and "noise" arrays.
   - The RMS power of the speech is divided by the RMS power of the silence to calculate Signal-to-Noise Ratio (SNR) in Decibels.
   - If SNR is too low, it fails as `NOISY_AUDIO`.

4. **Loudness & Clipping (`loudness_clipping.py`)**
   - Checks for digital clipping (amplitudes hitting 0.99 ceiling).
   - Uses `pyloudnorm` to calculate LUFS.

5. **Background Music Detection (`music_detection.py`)**
   - The spoken regions are sliced out and stitched together.
   - This "speech-only" tape is fed to **PANNs CNN14**.
   - If the output probability for AudioSet class `137` (Music) exceeds `0.40`, it fails.

6. **Overlap Detection (`overlap_detection.py`)**
   - Attempts to find polyphony (two people talking at exactly the same time).
   - If Pyannote is not configured, relies on an STFT spectral-peak heuristic looking for interlaced harmonics.

7. **Multiple Speakers (`speaker_diarization.py`)**
   - Slices the audio into 2-second chunks.
   - Computes a speaker embedding (d-vector) for each chunk using **ECAPA-TDNN**.
   - Runs Agglomerative Clustering (Cosine distance). If multiple clusters exist, it fails as `MULTIPLE_SPEAKERS_DETECTED`.

8. **Clarity & Pace (`transcription_asr.py`, `speech_pace.py`)**
   - The full audio is transcribed by **Faster-Whisper (Tiny)**.
   - If the average log probability of the transcription is too low, the speech is deemed unintelligible.
   - The duration of the words is measured to compute Pace (Words Per Minute).

9. **Tone Consistency (`tone_analysis.py`)**
   - Generates a fundamental frequency (f0) pitch contour using `librosa.pyin`.
   - Checks for extreme standard deviation (monotone) or massive spikes (spliced audio).

10. **Response Assembly**
    - The JSON dictionary is sanitized (to strip `numpy.float32` native objects) and returned to the client.
