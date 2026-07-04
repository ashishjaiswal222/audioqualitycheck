import os
import tempfile
from app.utils.audio_io import load_and_normalize_audio
from app.verification.vad_silence import detect_speech_regions, check_speech_presence
from app.verification.noise_snr import check_noise_snr
from app.verification.loudness_clipping import check_loudness_and_clipping
from app.verification.music_detection import check_background_music
from app.verification.overlap_detection import check_overlap
from app.verification.speaker_diarization import check_multiple_speakers, get_audio_embedding
from app.verification.transcription_asr import transcribe_and_check_clarity
from app.verification.speech_pace import check_speech_pace
from app.verification.tone_analysis import check_tone_consistency
from app.verification.models_loader import get_models

def run_audio_verification(file_bytes: bytes, filename: str, strict_tone: bool = False) -> dict:
    models = get_models()
    
    # 1. I/O & Normalize
    audio, duration = load_and_normalize_audio(file_bytes, filename)
    
    # 2. VAD
    speech_regions = detect_speech_regions(audio, models["vad"], models["vad_utils"])
    vad_pass, speech_prop, vad_reason, vad_msg, vad_details = check_speech_presence(speech_regions, duration)
    
    checks = {}
    checks["silence_presence"] = {"passed": vad_pass, "score": speech_prop}
    
    if not vad_pass:
        return _fail_result(vad_reason, vad_msg, duration, checks)

    # 3. Noise / SNR
    noise_pass, snr_db, noise_reason, noise_msg, noise_details = check_noise_snr(audio, speech_regions, duration)
    checks["noise"] = {"passed": noise_pass, "score": round(snr_db, 2), "unit": "dB_SNR"}
    if not noise_pass:
        return _fail_result(noise_reason, noise_msg, duration, checks)

    # 4. Loudness / Clipping
    loud_pass, lufs, loud_reason, loud_msg, loud_details = check_loudness_and_clipping(audio)
    checks["loudness_clipping"] = {"passed": loud_pass, **loud_details}
    if not loud_pass:
        return _fail_result(loud_reason, loud_msg, duration, checks)

    # 5. Background Music
    music_pass, music_conf, music_reason, music_msg, music_details = check_background_music(audio, speech_regions, models["panns"])
    checks["background_music"] = {"passed": music_pass, "score": round(music_conf, 2)}
    if not music_pass:
        return _fail_result(music_reason, music_msg, duration, checks)

    # 6. Overlapping Speech
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        import soundfile as sf
        sf.write(tmp.name, audio, 16000)
        tmp_name = tmp.name
    
    try:
        overlap_pass, overlap_sec, overlap_reason, overlap_msg, overlap_details = check_overlap(tmp_name, audio, models["pyannote"])
    finally:
        os.remove(tmp_name)
        
    checks["overlapping_voices"] = {"passed": overlap_pass, **overlap_details}
    if not overlap_pass:
        return _fail_result(overlap_reason, overlap_msg, duration, checks)

    # 7. Multiple Speakers
    spk_pass, num_clusters, cluster_shares, spk_reason, spk_msg, spk_details = check_multiple_speakers(audio, speech_regions, models["ecapa"])
    checks["multiple_speakers"] = {"passed": spk_pass, **spk_details}
    if not spk_pass:
        return _fail_result(spk_reason, spk_msg, duration, checks)

    # 8. Transcription & Clarity
    asr_pass, avg_conf, langs, words, asr_reason, asr_msg, asr_details = transcribe_and_check_clarity(audio, models["whisper"])
    checks["speech_clarity"] = {"passed": asr_pass, **asr_details}
    if not asr_pass:
        return _fail_result(asr_reason, asr_msg, duration, checks, languages_detected=langs)

    # 9. Pace
    pace_pass, wpm, pace_reason, pace_msg, pace_details = check_speech_pace(words, duration)
    checks["speech_pace"] = {"passed": pace_pass, **pace_details}
    if not pace_pass:
        return _fail_result(pace_reason, pace_msg, duration, checks, languages_detected=langs)

    # 10. Tone
    tone_pass, tone_reason, tone_msg, tone_details = check_tone_consistency(audio, speech_regions, strict_tone)
    checks["tone_consistency"] = {"passed": tone_pass, **tone_details}
    if not tone_pass:
        return _fail_result(tone_reason, tone_msg, duration, checks, languages_detected=langs)

    embedding = get_audio_embedding(audio, models["ecapa"]) if models["ecapa"] else None

    return {
        "passed": True,
        "confidence": checks["speech_clarity"]["avg_confidence"],
        "duration_seconds": round(duration, 1),
        "languages_detected": langs,
        "primary_reason": None,
        "checks": checks,
        "embedding": embedding
    }

def _fail_result(code: str, message: str, duration: float, checks: dict, languages_detected: list = None):
    return {
        "passed": False,
        "confidence": 0.0,
        "duration_seconds": round(duration, 1),
        "languages_detected": languages_detected or [],
        "primary_reason": {
            "code": code,
            "message": message
        },
        "checks": checks,
        "embedding": None
    }
