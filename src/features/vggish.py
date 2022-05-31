from typing import Optional
import numpy as np
from .utils import load_audio, pooling, GlobalPooling

vggish: Optional = None


def get_vggish(*args, **kwargs):
    global vggish
    if vggish is None:
        ...

    return vggish


def get_vggish_features(
        file_path: str,
        *model_args,
        t_pooling: Optional[GlobalPooling] = None,
        tgt_len: Optional[float] = None,
        **model_kwargs
) -> np.ndarray:
    raw_data, sample_rate = load_audio(file_path, tgt_len=tgt_len)
    model = get_vggish(*model_args, **model_kwargs)

    audio_features = ...

    if pooling is not None:
        return pooling(audio_features, t_pooling)
    else:
        return audio_features
