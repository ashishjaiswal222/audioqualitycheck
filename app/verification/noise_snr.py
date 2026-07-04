import numpy as np
import math
from app.verification.constants import ReasonCode, MIN_SNR_DB, TARGET_SAMPLE_RATE

def check_noise_snr(audio: np.ndarray, speech_regions: list[dict], duration: float) -> tuple[bool, float, str, str, dict]:
    """
    Estimates SNR using speech regions vs non-speech regions.
    """
    # Create a boolean mask for speech
    is_speech = np.zeros(len(audio), dtype=bool)
    for r in speech_regions:
        start_idx = int(r["start"] * TARGET_SAMPLE_RATE)
        end_idx = int(r["end"] * TARGET_SAMPLE_RATE)
        is_speech[start_idx:end_idx] = True

    speech_samples = audio[is_speech]
    noise_samples = audio[~is_speech]

    if len(speech_samples) == 0:
        return False, 0.0, ReasonCode.SILENCE_ONLY.value, "Audio contains no speech.", {}

    if len(noise_samples) < TARGET_SAMPLE_RATE * 0.1:  # less than 100ms of silence
        snr_db = 50.0  # assume good SNR if not enough silence to measure
    else:
        speech_power = np.mean(speech_samples**2)
        noise_power = np.mean(noise_samples**2)
        
        if noise_power < 1e-10:
            snr_db = 50.0
        else:
            snr_db = 10 * math.log10(speech_power / noise_power)

    passed = snr_db >= MIN_SNR_DB
    
    reason_code = ReasonCode.NOISY_AUDIO.value if not passed else None
    message = "Background noise detected. Please record in a quiet environment." if not passed else None
    
    return passed, snr_db, reason_code, message, {"snr_db": snr_db}
