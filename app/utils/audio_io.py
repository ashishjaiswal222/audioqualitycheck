import os
import io
import soundfile as sf
import librosa
import pydub
import numpy as np
from app.utils.errors import AudioVerificationError
from app.verification.constants import (
    ReasonCode, MAX_FILE_SIZE_BYTES, MIN_DURATION_SECONDS, 
    MAX_DURATION_SECONDS, TARGET_SAMPLE_RATE
)

def load_and_normalize_audio(file_bytes: bytes, filename: str) -> tuple[np.ndarray, float]:
    """
    Loads audio from bytes, validates format/size/duration, and normalizes
    to mono, 16kHz float32 numpy array.
    """
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise AudioVerificationError(
            code=ReasonCode.FILE_TOO_LARGE.value,
            message=f"File exceeds maximum size of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB."
        )

    ext = os.path.splitext(filename)[1].lower().lstrip('.')
    if ext not in ["wav", "mp3", "m4a", "ogg", "flac"]:
        raise AudioVerificationError(
            code=ReasonCode.UNSUPPORTED_FORMAT.value,
            message=f"Unsupported format: {ext}. Allowed: wav, mp3, m4a, ogg, flac."
        )

    try:
        if ext in ["wav", "flac", "ogg"]:
            # Soundfile handles these natively without ffmpeg
            audio_data, sr = sf.read(io.BytesIO(file_bytes))
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1) # convert to mono
        else:
            # pydub requires ffmpeg for mp3, m4a
            audio_seg = pydub.AudioSegment.from_file(io.BytesIO(file_bytes), format=ext)
            audio_seg = audio_seg.set_channels(1) # mono
            samples = np.array(audio_seg.get_array_of_samples())
            
            # Normalize to float32 between -1.0 and 1.0
            if audio_seg.sample_width == 2:
                audio_data = samples.astype(np.float32) / 32768.0
            elif audio_seg.sample_width == 4:
                audio_data = samples.astype(np.float32) / 2147483648.0
            else:
                # 8-bit or other
                audio_data = samples.astype(np.float32) / (2**(audio_seg.sample_width * 8 - 1))
            
            sr = audio_seg.frame_rate

        # Ensure float32 (soundfile might return float64)
        audio_data = audio_data.astype(np.float32)

        # Resample if needed
        if sr != TARGET_SAMPLE_RATE:
            audio_data = librosa.resample(y=audio_data, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
            sr = TARGET_SAMPLE_RATE

    except Exception as e:
        raise AudioVerificationError(
            code=ReasonCode.CORRUPT_AUDIO.value,
            message="The audio file is corrupt or could not be decoded."
        )

    duration = len(audio_data) / sr

    if duration < MIN_DURATION_SECONDS:
        raise AudioVerificationError(
            code=ReasonCode.DURATION_TOO_SHORT.value,
            message=f"Audio is too short ({duration:.1f}s). Minimum is {MIN_DURATION_SECONDS}s."
        )
    if duration > MAX_DURATION_SECONDS:
        raise AudioVerificationError(
            code=ReasonCode.DURATION_TOO_LONG.value,
            message=f"Audio is too long ({duration:.1f}s). Maximum is {MAX_DURATION_SECONDS}s."
        )

    return audio_data, duration
