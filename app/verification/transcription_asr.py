import numpy as np
from app.verification.constants import ReasonCode, MIN_AVG_LOGPROB

def transcribe_and_check_clarity(audio: np.ndarray, whisper_model) -> tuple[bool, float, list, list, str, str, dict]:
    """
    Transcribes audio, checks clarity via avg_logprob.
    """
    if not whisper_model:
        return True, 0.0, [], [], None, None, {}

    # Faster Whisper accepts numpy arrays directly
    segments, info = whisper_model.transcribe(audio, beam_size=5, word_timestamps=True)
    
    segments = list(segments)
    
    if not segments:
        return True, 0.0, [], [], None, None, {}
        
    total_dur = 0
    weighted_logprob = 0
    languages_detected = set()
    all_words = []
    
    for s in segments:
        dur = s.end - s.start
        total_dur += dur
        weighted_logprob += s.avg_logprob * dur
        languages_detected.add(info.language)
        if s.words:
            all_words.extend(s.words)
            
    avg_confidence = weighted_logprob / total_dur if total_dur > 0 else 0
    passed = avg_confidence >= MIN_AVG_LOGPROB
    
    reason_code = ReasonCode.SPEECH_UNCLEAR.value if not passed else None
    message = "Speech is unclear or mumbled in parts. Please speak clearly and directly into the microphone." if not passed else None
    
    return passed, avg_confidence, list(languages_detected), all_words, reason_code, message, {
        "avg_confidence": round(avg_confidence, 2)
    }
