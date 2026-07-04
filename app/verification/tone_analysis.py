import numpy as np
import librosa
from app.verification.constants import ReasonCode, TARGET_SAMPLE_RATE

def check_tone_consistency(audio: np.ndarray, speech_regions: list[dict], strict: bool = False) -> tuple[bool, str, str, dict]:
    """
    Soft check for tone naturalness via pitch contour.
    1. Checks if pitch is unusually flat (monotone).
    2. Checks for abrupt pitch-range shifts (potential splice).
    """
    if not speech_regions:
        return True, None, None, {"advisory": True}
        
    try:
        f0, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=TARGET_SAMPLE_RATE)
        f0 = f0[~np.isnan(f0)]
        
        if len(f0) == 0:
            return True, None, None, {"advisory": True}
            
        std_dev = np.std(f0)
        
        # 1. Monotone check
        is_monotone = std_dev < 10.0
        
        if is_monotone:
            reason = ReasonCode.TONE_INCONSISTENT.value
            msg = "Speech pitch is unusually flat (monotone)."
            if strict:
                return False, reason, msg, {"advisory": False, "pitch_std": float(std_dev)}
            return True, reason, msg, {"advisory": True, "pitch_std": float(std_dev)}
            
        # 2. Pitch shift check (splice detection)
        # Hop length for pyin defaults to 512, which is ~31 frames per second.
        # 2 seconds = ~62 frames
        window_size = 62
        if len(f0) > window_size * 2: 
            # Compute moving average of pitch
            window = np.ones(window_size) / window_size
            f0_smooth = np.convolve(f0, window, mode='valid')
            
            # Find the jump between adjacent blocks
            pitch_gradient = np.abs(np.diff(f0_smooth))
            max_shift = np.max(pitch_gradient)
            
            # 50Hz sustained shift between windows is highly unnatural for a single speaker
            if max_shift > 50.0:
                reason = ReasonCode.TONE_INCONSISTENT.value
                msg = "Abrupt pitch shift detected. Potential edit or splice."
                if strict:
                    return False, reason, msg, {"advisory": False, "pitch_shift_hz": float(max_shift)}
                return True, reason, msg, {"advisory": True, "pitch_shift_hz": float(max_shift)}

    except Exception:
        pass
        
    return True, None, None, {"advisory": True}
