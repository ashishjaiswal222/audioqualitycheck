import numpy as np
import torch
from app.verification.constants import ReasonCode, MIN_SPEECH_PROPORTION

def detect_speech_regions(audio: np.ndarray, vad_model, vad_utils) -> list[dict]:
    """
    Returns list of speech regions [{"start": float, "end": float}] in seconds.
    vad_utils is the tuple returned by Silero's torch.hub load.
    """
    (get_speech_timestamps, _, _, _, _) = vad_utils
    
    # Silero requires torch tensor
    tensor = torch.from_numpy(audio).float()
    
    # Get timestamps in samples
    speech_timestamps = get_speech_timestamps(tensor, vad_model, sampling_rate=16000)
    
    # Convert to seconds
    regions = []
    for ts in speech_timestamps:
        regions.append({
            "start": ts["start"] / 16000.0,
            "end": ts["end"] / 16000.0
        })
    return regions

def check_speech_presence(regions: list[dict], duration: float) -> tuple[bool, float, str, str, dict]:
    """
    Validates if the audio has enough speech.
    """
    if duration == 0:
        return False, 0.0, ReasonCode.SILENCE_ONLY.value, "Audio is completely silent.", {}
        
    total_speech = sum(r["end"] - r["start"] for r in regions)
    proportion = total_speech / duration
    
    if proportion == 0:
        return False, proportion, ReasonCode.SILENCE_ONLY.value, "This recording doesn't contain enough spoken audio. Please re-record with clear, continuous speech.", {"speech_proportion": proportion}
        
    if proportion < MIN_SPEECH_PROPORTION:
        return False, proportion, ReasonCode.INSUFFICIENT_SPEECH.value, "This recording doesn't contain enough spoken audio. Please re-record with clear, continuous speech.", {"speech_proportion": proportion}
        
    return True, proportion, None, None, {"speech_proportion": proportion}
