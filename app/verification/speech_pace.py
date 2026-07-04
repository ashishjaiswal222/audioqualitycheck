from app.verification.constants import ReasonCode, MIN_WORDS_PER_MINUTE, MAX_WORDS_PER_MINUTE

def check_speech_pace(words: list, duration: float) -> tuple[bool, float, str, str, dict]:
    """
    Calculates words per minute.
    """
    if not words or duration == 0:
        return True, 0.0, None, None, {}

    first_word_start = words[0].start
    last_word_end = words[-1].end
    speaking_duration = last_word_end - first_word_start
    
    if speaking_duration <= 0:
        return True, 0.0, None, None, {}
        
    wpm = (len(words) / speaking_duration) * 60.0
    
    passed = True
    reason_code = None
    message = None
    
    if wpm < MIN_WORDS_PER_MINUTE:
        passed = False
        reason_code = ReasonCode.SPEECH_TOO_SLOW.value
        message = "Speech pace is too slow. Please speak at a natural, steady pace."
    elif wpm > MAX_WORDS_PER_MINUTE:
        reason_code = ReasonCode.SPEECH_TOO_FAST.value
        message = "Speech pace is quite fast."
        
    return passed, wpm, reason_code, message, {"words_per_minute": round(wpm)}
