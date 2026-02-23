import numpy as np
import pandas as pd
import librosa
import io
import soundfile as sf


# Exact feature order matching the training data
FEATURE_NAMES = [
    'chroma_stft_mean', 'chroma_stft_var',
    'rms_mean', 'rms_var',
    'spectral_centroid_mean', 'spectral_centroid_var',
    'spectral_bandwidth_mean', 'spectral_bandwidth_var',
    'rolloff_mean', 'rolloff_var',
    'zero_crossing_rate_mean', 'zero_crossing_rate_var',
    'harmony_mean', 'harmony_var',
    'perceptr_mean', 'perceptr_var',
    'tempo',
]
# Add MFCC 1-20 (mean and var)
for i in range(1, 21):
    FEATURE_NAMES.extend([f'mfcc{i}_mean', f'mfcc{i}_var'])

assert len(FEATURE_NAMES) == 57, f"Expected 57 features, got {len(FEATURE_NAMES)}"


def extract_features_from_file(file_path: str, duration: float = 30.0) -> np.ndarray:
    """
    Extract features from an audio file on disk.

    Args:
        file_path: Path to the audio file (wav, mp3, ogg, flac, etc.)
        duration: Duration in seconds to analyze (default: 30s to match training data)

    Returns:
        numpy array of shape (1, 57) — one row of features
    """
    y, sr = librosa.load(file_path, duration=duration, sr=22050)
    return _extract(y, sr)


def extract_features_from_bytes(audio_bytes: bytes, duration: float = 30.0) -> np.ndarray:
    """
    Extract features from raw audio bytes (e.g. from browser MediaRecorder).
    Supports webm, ogg, wav, and any format soundfile/librosa can decode.

    Args:
        audio_bytes: Raw audio file bytes
        duration: Duration in seconds to analyze

    Returns:
        numpy array of shape (1, 57)
    """
    # Try loading via soundfile first (handles wav, ogg, flac)
    try:
        buf = io.BytesIO(audio_bytes)
        y, sr = sf.read(buf, dtype='float32')
        # Convert to mono if stereo
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        # Resample to 22050 if needed
        if sr != 22050:
            y = librosa.resample(y, orig_sr=sr, target_sr=22050)
            sr = 22050
        # Trim to duration
        max_samples = int(duration * sr)
        y = y[:max_samples]
    except Exception:
        # Fallback: write to temp file and let librosa handle it (for webm, mp3, etc.)
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            y, sr = librosa.load(tmp_path, duration=duration, sr=22050)
        finally:
            os.unlink(tmp_path)

    return _extract(y, sr)


def _extract(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Core feature extraction from a waveform array.

    Args:
        y: Audio time series (mono, float32)
        sr: Sample rate

    Returns:
        numpy array of shape (1, 57)
    """
    features = []

    # 1. Chroma STFT
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.append(chroma.mean())
    features.append(chroma.var())

    # 2. RMS Energy
    rms = librosa.feature.rms(y=y)
    features.append(rms.mean())
    features.append(rms.var())

    # 3. Spectral Centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(spec_cent.mean())
    features.append(spec_cent.var())

    # 4. Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(spec_bw.mean())
    features.append(spec_bw.var())

    # 5. Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(rolloff.mean())
    features.append(rolloff.var())

    # 6. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(zcr.mean())
    features.append(zcr.var())

    # 7. Harmony & Percussive
    harmony, perceptr = librosa.effects.hpss(y)
    features.append(harmony.mean())
    features.append(harmony.var())
    features.append(perceptr.mean())
    features.append(perceptr.var())

    # 8. Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # librosa >= 0.10 returns an array; extract scalar
    if isinstance(tempo, np.ndarray):
        tempo = tempo.item()
    features.append(float(tempo))

    # 9. MFCCs 1-20
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features.append(mfccs[i].mean())
        features.append(mfccs[i].var())

    return pd.DataFrame([features], columns=FEATURE_NAMES)


def get_system_audio_monitor() -> str:
    """
    Auto-detect the PulseAudio/PipeWire monitor source for system audio output.
    Returns the source name (e.g. 'alsa_output....sink.monitor').
    """
    import subprocess
    result = subprocess.run(
        ['pactl', 'list', 'short', 'sources'],
        capture_output=True, text=True, timeout=5
    )
    if result.returncode != 0:
        raise RuntimeError("pactl not available. Is PulseAudio/PipeWire running?")

    # Look for a .monitor source (loopback of an output sink)
    for line in result.stdout.strip().split('\n'):
        parts = line.split('\t')
        if len(parts) >= 2 and '.monitor' in parts[1]:
            # Prefer speaker/headphone monitor over HDMI
            if 'hdmi' not in parts[1].lower():
                return parts[1]

    # Fallback: return any monitor source
    for line in result.stdout.strip().split('\n'):
        parts = line.split('\t')
        if len(parts) >= 2 and '.monitor' in parts[1]:
            return parts[1]

    raise RuntimeError("No monitor source found. Cannot capture system audio.")


def extract_features_from_system_audio(duration: float = 10.0) -> np.ndarray:
    """
    Capture system audio (whatever is currently playing through speakers)
    and extract features from it.

    Uses PulseAudio/PipeWire's `parec` to record from the monitor source.

    Args:
        duration: Seconds of audio to capture (default: 10s)

    Returns:
        numpy array of shape (1, 57)

    Raises:
        RuntimeError: If no audio system is available or no audio is playing
    """
    import subprocess

    monitor = get_system_audio_monitor()
    sample_rate = 22050
    channels = 1

    # Record raw PCM audio from the system monitor source
    cmd = [
        'parec',
        '--device', monitor,
        '--rate', str(sample_rate),
        '--channels', str(channels),
        '--format', 's16le',      # 16-bit signed little-endian
        '--raw',
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Read for the specified duration
    num_bytes = int(duration * sample_rate * channels * 2)  # 2 bytes per sample (s16le)
    raw_audio = proc.stdout.read(num_bytes)
    proc.terminate()
    proc.wait()

    if len(raw_audio) < sample_rate * 2:  # less than 1 second captured
        raise RuntimeError("Too little audio captured. Is something playing?")

    # Convert raw PCM bytes to numpy float array
    y = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
    y = y / 32768.0  # normalize to [-1.0, 1.0]

    return _extract(y, sample_rate)
