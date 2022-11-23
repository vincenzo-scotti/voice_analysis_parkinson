from typing import Optional, Tuple, List
import numpy as np
import librosa
from enum import Enum
import math


class GlobalPooling(Enum):
    AVERAGE: str = 'avg'
    MAXIMUM: str = 'max'
    FLATTENING: str = 'flatten'


def load_audio(file_path: str, tgt_len: Optional[float] = None, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    raw_audio_data, sample_rate = librosa.load(file_path, sr=sr)
    if tgt_len is not None and tgt_len > 0.0:
        padding = int((tgt_len * sample_rate) - len(raw_audio_data))
        raw_audio_data = np.pad(raw_audio_data, pad_width=(0, padding))

    return raw_audio_data, sample_rate


# TODO add function chucking to split the audio
def trunc_audio(x, y, d, chunk_len=4.0) -> List[Tuple[np.ndarray,np.ndarray]]:
    list_chunks =[]
    windows_number = int(math.ceil(d / chunk_len))
    #durata totale : #campioni_totali = durata chunk  : #campioni_chunk
    sample_number = x.shape[0]
    chunk_sample_number = int((chunk_len * sample_number) // d)
    # 'median' o 'edge' (entrambe le direzioni)

    #split matrix of features, chunk starts from 0
    for chunk in range(0,windows_number):
        start_x = chunk * chunk_sample_number
        if chunk == windows_number-1:
            #padding part
            last_chunk = x[start_x:, :]
            diff =chunk_sample_number-(sample_number % chunk_sample_number)
            #TODO check if .T is needed
            last_chunk = np.pad(last_chunk, ((diff//2, math.ceil(diff/2)),(0,0)),'edge')
            list_chunks.append(last_chunk)
        else:
            list_chunks.append(x[start_x:start_x+chunk_sample_number,:])

    final_list_chunks = [(elem, y) for elem in list_chunks]
    return final_list_chunks



def pooling(data: np.ndarray, t_pooling: GlobalPooling) -> np.ndarray:
    if t_pooling == GlobalPooling.AVERAGE:
        return np.nanmean(data, axis=0)
    elif t_pooling == GlobalPooling.MAXIMUM:
        return np.nanmax(data, axis=0)
    else:
        data = np.nan_to_num(data)
        # TODO do resampling
        return data.reshape(-1)
