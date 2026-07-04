import numpy as np
import pyloudnorm
from app.verification.constants import ReasonCode, MIN_LUFS, TARGET_SAMPLE_RATE

def check_loudness_and_clipping(audio: np.ndarray) -> tuple[bool, float, str, str, dict]:
    """
    Checks for clipping and loudness via pyloudnorm.
    """
    # 1. Clipping detection
    clipping_ratio = float(np.sum(np.abs(audio) >= 0.99) / len(audio))
    
    if clipping_ratio > 0.01:
        return False, clipping_ratio, ReasonCode.AUDIO_CLIPPING_DETECTED.value, "Audio clipping detected. Please reduce your microphone volume.", {"clipping_ratio": clipping_ratio}

    # 2. Loudness check
    meter = pyloudnorm.Meter(TARGET_SAMPLE_RATE)
    try:
        lufs = float(meter.integrated_loudness(audio))
    except ValueError:
        lufs = -70.0 # fallback if it fails (e.g. silence)
        
    if lufs < MIN_LUFS:
        return False, lufs, ReasonCode.LOW_VOLUME.value, "Audio volume is too low. Please speak closer to the microphone or increase volume.", {"lufs": lufs}

    return True, lufs, None, None, {"lufs": lufs, "clipping_ratio": clipping_ratio}
