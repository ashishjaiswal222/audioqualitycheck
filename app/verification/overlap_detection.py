import numpy as np
import librosa
import scipy.signal
from app.verification.constants import ReasonCode, MAX_OVERLAP_TOLERANCE_SECONDS, TARGET_SAMPLE_RATE
from app.config import settings

def _heuristic_overlap_fallback(audio: np.ndarray) -> float:
    """
    A basic heuristic for overlapped speech (polyphony) using spectral peak analysis.
    Looks for frames with an unusually high number of prominent frequency peaks 
    in the typical speech fundamental range, which suggests multiple voices.
    """
    stft_mag = np.abs(librosa.stft(audio))
    # sr=16000, n_fft=2048, bin width ~7.8Hz.
    # Typical human fundamental frequencies are 85Hz to 255Hz, 
    # and their first few harmonics up to ~1500Hz.
    # Let's analyze 85Hz (bin 11) to 1500Hz (bin 192).
    speech_band = stft_mag[11:192, :]
    
    overlap_frames = 0
    total_frames = speech_band.shape[1]
    
    for i in range(total_frames):
        frame = speech_band[:, i]
        max_val = np.max(frame)
        if max_val < 0.1:  # Skip quiet frames
            continue
            
        # Find peaks with a prominence threshold
        # Overlapping voices produce interlaced harmonic structures (more peaks)
        peaks, _ = scipy.signal.find_peaks(frame, prominence=0.2 * max_val)
        
        # A single voice usually has 3-6 clear harmonics in this band. 
        # More than 8 suggests polyphony/overlap.
        if len(peaks) > 8:
            overlap_frames += 1
            
    # Standard librosa STFT hop length is 512
    overlap_seconds = overlap_frames * (512 / TARGET_SAMPLE_RATE)
    return overlap_seconds

def check_overlap(audio_file_path: str, audio_array: np.ndarray, pyannote_pipeline) -> tuple[bool, float, str, str, dict]:
    """
    Checks for overlapping speech (crosstalk).
    """
    overlap_seconds = 0.0
    fallback_used = False
    
    if settings.use_overlap_heuristic_fallback or not pyannote_pipeline:
        overlap_seconds = _heuristic_overlap_fallback(audio_array)
        fallback_used = True
    else:
        try:
            overlap_annotation = pyannote_pipeline(audio_file_path)
            for segment in overlap_annotation.itersegments():
                overlap_seconds += segment.end - segment.start
        except Exception:
            # If pyannote fails at runtime, fallback silently to heuristic
            overlap_seconds = _heuristic_overlap_fallback(audio_array)
            fallback_used = True

    passed = overlap_seconds <= MAX_OVERLAP_TOLERANCE_SECONDS
    
    reason_code = ReasonCode.OVERLAPPING_VOICES_DETECTED.value if not passed else None
    message = "Two people appear to be speaking at the same time. Please provide a recording of a single speaker with no crosstalk." if not passed else None
    
    return passed, overlap_seconds, reason_code, message, {
        "overlap_seconds": round(overlap_seconds, 2),
        "heuristic_fallback_used": fallback_used
    }
