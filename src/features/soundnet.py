from typing import Optional
from .utils import load_audio, pooling


def get_soundnet_features(file_path: str, t_pooling: Optional[str] = None, tgt_len: Optional[float] = None):
    data = load_audio(file_path, tgt_len=tgt_len)
    ...
    if pooling is not None:
        return pooling(data, t_pooling)
    else:
        return data
