from typing import Optional
from .utils import load_audio, pooling, GlobalPooling


def get_spectral_features(file_path: str, t_pooling: Optional[GlobalPooling] = None, tgt_len: Optional[int] = None):
    data, sample_rate = load_audio(file_path, tgt_len=tgt_len)
    ...
    if pooling is not None:
        return pooling(data, t_pooling)
    else:
        return data
