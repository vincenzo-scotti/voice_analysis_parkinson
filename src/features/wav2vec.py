from typing import Optional
import numpy as np
from .utils import load_audio, pooling, GlobalPooling

wav2vec: Optional = None


def get_wav2vec(*args, **kwargs):
    global wav2vec
    if wav2vec is None:
        ...

    return wav2vec


def get_wav2vec_features(
        file_path: str,
        *model_args,
        t_pooling: Optional[GlobalPooling] = None,
        tgt_len: Optional[float] = None,
        **model_kwargs
) -> np.ndarray:
    raw_data, sample_rate = load_audio(file_path, tgt_len=tgt_len)
    model = get_wav2vec(*model_args, **model_kwargs)

    audio_features = ...

    if pooling is not None:
        return pooling(audio_features, t_pooling)
    else:
        return audio_features
