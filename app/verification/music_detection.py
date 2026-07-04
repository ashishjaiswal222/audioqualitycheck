import numpy as np
from app.verification.constants import ReasonCode, MUSIC_CLASS_THRESHOLD, TARGET_SAMPLE_RATE

def check_background_music(audio: np.ndarray, speech_regions: list[dict], panns_model) -> tuple[bool, float, str, str, dict]:
    """
    Checks for music during speech using PANNs.
    """
    if not speech_regions or not panns_model:
        return True, 0.0, None, None, {}

    # Gather audio from speech regions
    speech_audio = []
    for r in speech_regions:
        start = int(r["start"] * TARGET_SAMPLE_RATE)
        end = int(r["end"] * TARGET_SAMPLE_RATE)
        speech_audio.extend(audio[start:end])
        
    if len(speech_audio) == 0:
        return True, 0.0, None, None, {}
        
    speech_audio = np.array(speech_audio, dtype=np.float32)
    
    # Inference batch shape: (1, num_samples)
    batch = speech_audio[None, :] 
    
    (clipwise_output, _) = panns_model.inference(batch)
    
    # 137 is Music in AudioSet, we also check related ones
    try:
        from panns_inference import labels
        music_indices = [i for i, label in enumerate(labels) if "music" in label.lower()]
        max_music_prob = max([clipwise_output[0][i] for i in music_indices])
    except Exception:
        # Fallback to hardcoded index 137 if labels can't be imported
        max_music_prob = clipwise_output[0][137]
    
    if max_music_prob > MUSIC_CLASS_THRESHOLD:
        return False, float(max_music_prob), ReasonCode.BACKGROUND_MUSIC_DETECTED.value, "Background music detected. Please provide a voice-only recording with no music.", {"music_confidence": float(max_music_prob)}
        
    return True, float(max_music_prob), None, None, {"music_confidence": float(max_music_prob)}
