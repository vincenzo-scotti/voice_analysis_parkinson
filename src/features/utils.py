from typing import Optional, Tuple, List
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


# TODO add function chucking to split the audio
# 3 input lists: 1 feature audio matrix; 2 list with same lngt of 1st with labels
#  3 (same length) #of chunks in output
# padding to have same length
def trunc_audio(x, y, d, chunk_len=4.0) -> List[Tuple[np.ndarray, np.ndarray]]:
    # calcolo# finestre  = int(math.ceil(d / chunk_len)
    # 9,6 sec : #campioni = 4 sec : #campioni(nuovi)
    # 'median' o 'edge' (entrambe le direzioni)
    #shape[0] lunghezza in campioni del file audio
    # dur (sec) : fin (sec) = dur (campioni) : fin (campioni)
    # ripeter label tante volte quante sono le finestre
    list_chunks = ...
    return list_chunks


def pooling(data: np.ndarray, t_pooling: GlobalPooling) -> np.ndarray:
    if t_pooling == GlobalPooling.AVERAGE:
        return np.nanmean(data, axis=0)
    elif t_pooling == GlobalPooling.MAXIMUM:
        return np.nanmax(data, axis=0)
    else:
        data = np.nan_to_num(data)
        # TODO do resampling
        return data.reshape(-1)
