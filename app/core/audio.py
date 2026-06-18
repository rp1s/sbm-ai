from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def extract_wav_features(
    path: Path | None = None,
    data: bytes | None = None,
    target_dim: int = 32,
) -> Array:
    if data is None and path is None:
        raise ValueError("Нужно передать путь или байты аудио")

    if data is None:
        with open(path, "rb") as f:
            data = f.read()

    with wave.open(io.BytesIO(data), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)

    if sample_width == 1:
        dtype = np.uint8
        offset = 128
    elif sample_width == 2:
        dtype = np.int16
        offset = 0
    else:
        raise ValueError(f"Неподдерживаемая разрядность WAV: {sample_width}")

    signal = np.frombuffer(frames, dtype=dtype).astype(np.float64)
    if offset:
        signal -= offset

    if n_channels > 1:
        signal = signal.reshape(-1, n_channels).mean(axis=1)

    if signal.size == 0:
        return np.zeros(target_dim, dtype=np.float64)

    time_dim = target_dim // 2
    spec_dim = target_dim - time_dim

    blocks = np.array_split(signal, time_dim)
    time_features = np.array(
        [np.mean(np.abs(block)) if block.size else 0.0 for block in blocks],
        dtype=np.float64,
    )

    n_fft = max(256, 1 << ((signal.size - 1).bit_length()))
    n_fft = min(4096, n_fft)

    if signal.size < n_fft:
        padded = np.zeros(n_fft, dtype=np.float64)
        padded[: signal.size] = signal * np.hanning(signal.size)
    else:
        padded = (signal[:n_fft] * np.hanning(n_fft)).astype(np.float64)

    spectrum = np.abs(np.fft.rfft(padded, n=n_fft))

    if spec_dim > 0:
        edges = np.linspace(0, spectrum.size, spec_dim + 1, dtype=int)
        spec_features = np.array(
            [
                np.mean(spectrum[edges[i] : edges[i + 1]]) if edges[i] < edges[i + 1] else 0.0
                for i in range(spec_dim)
            ],
            dtype=np.float64,
        )
    else:
        spec_features = np.zeros(0, dtype=np.float64)

    features = np.concatenate([time_features, spec_features])

    if np.max(features) > 0:
        features /= np.max(features)

    return features


def extract_audio_features(
    path: Path,
    target_dim: int = 32,
) -> Array:
    suffix = path.suffix.lower()

    if suffix == ".wav":
        return extract_wav_features(path=path, target_dim=target_dim)

    elif suffix in {".mp3", ".ogg", ".flac"}:
        try:
            import librosa
            y, sr = librosa.load(str(path), sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=target_dim)
            return np.mean(mfcc, axis=1)
        except ImportError:
            raise RuntimeError("Для MP3/OGG/FLAC нужен пакет librosa: pip install librosa")
        except Exception as e:
            print(f"Ошибка загрузки {path}: {e}")
            return np.zeros(target_dim, dtype=np.float64)

    else:
        raise ValueError(f"Неподдерживаемый формат аудио: {suffix}")
