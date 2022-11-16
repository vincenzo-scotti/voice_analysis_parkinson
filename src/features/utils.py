from typing import Optional, Tuple
import numpy as np
import librosa
from enum import Enum


class GlobalPooling(Enum):
    AVERAGE: str = 'avg'
    MAXIMUM: str = 'max'
    FLATTENING: str = 'flatten'


def get_audio_length(file_path: str) -> float:
    ...


def load_audio(file_path: str, tgt_len: Optional[float] = None, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    raw_audio_data, sample_rate = librosa.load(file_path, sr=sr)
    if tgt_len is not None and tgt_len > 0.0:
        padding = int((tgt_len * sample_rate) - len(raw_audio_data))
        raw_audio_data = np.pad(raw_audio_data, pad_width=(0, padding))

    return raw_audio_data, sample_rate


def pooling(data: np.ndarray, t_pooling: GlobalPooling) -> np.ndarray:
    if t_pooling == GlobalPooling.AVERAGE:
        return data.mean(axis=0)
    elif t_pooling == GlobalPooling.MAXIMUM:
        return data.max(axis=0)
    else:
        return data.reshape(-1)
