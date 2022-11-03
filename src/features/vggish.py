from typing import Optional
import numpy as np
from .utils import pooling, GlobalPooling
import torch

vggish: Optional = None


def get_vggish():
    global vggish
    if vggish is None:
        vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
    return vggish


def get_vggish_features(
        file_path: str,
        *model_args,
        t_pooling: Optional[GlobalPooling] = None,
        tgt_len: Optional[float] = None,
        **model_kwargs
) -> np.ndarray:
    model = get_vggish()
    
    #processing data
    audio_features = model.forward(file_path)

    if pooling is not None:
        return pooling(audio_features, t_pooling)
    else:
        return audio_features
