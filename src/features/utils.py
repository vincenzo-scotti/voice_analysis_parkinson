from typing import Optional, Tuple, List, Dict
import numpy as np
import librosa
from enum import Enum
import math
import os
import bz2
import pickle
import warnings


class GlobalPooling(Enum):
    AVERAGE: str = 'avg'
    MAXIMUM: str = 'max'
    FLATTENING: str = 'flatten'


class FeaturesCache:
    cache_data: Optional[Dict[str, np.ndarray]] = None
    cache_path: Optional[str] = None

    def __init__(self, cache_dir_path: str, feature_id: str):
        # Create cache dir if not exists
        if not os.path.exists(cache_dir_path):
            os.mkdir(cache_dir_path)
        # Identify specific features cache path
        cache_path = os.path.join(cache_dir_path, feature_id + '.pbz2')
        # Load cache if available otherwise generate it
        if self.cache_path is None or not self.cache_path == cache_path:
            # Set cache path static variable
            self.cache_path = cache_path
            # Manage cache mapping
            if os.path.exists(self.cache_path):
                # Load file if it exists
                self._load_cache_file()
            else:
                # Init cache data static variable
                self.cache_data = dict()
                # Create cache file
                self._update_cache_file()

    @staticmethod
    def get_cached_features(file_path: str) -> Optional[np.ndarray]:
        assert FeaturesCache.cache_data is not None
        # Get features from cache (may be None)
        return FeaturesCache.cache_data.get(file_path)

    @staticmethod
    def cache_features(file_path: str, features: np.ndarray):
        assert FeaturesCache.cache_data is not None and FeaturesCache.cache_path is not None
        # Store features in cache mapping
        FeaturesCache.cache_data[file_path] = features
        # Finally update the cache file
        FeaturesCache._update_cache_file()

    @staticmethod
    def _load_cache_file():
        with bz2.BZ2File(FeaturesCache.cache_path, 'r') as f:
            FeaturesCache.cache_data = pickle.load(f)

    @staticmethod
    def _update_cache_file():
        with bz2.BZ2File(FeaturesCache.cache_path, 'w') as f:
            pickle.dump(FeaturesCache.cache_data, f)


def safe_features_load(feature_loader):
    def wrapped_get_features(*args, **kwargs):
        try:
            return feature_loader(*args, **kwargs)
        except Exception as e:
            warnings.warn(f'During feature loading the following error occurred:\n\n{e}\n\nReturning null features.')
            return None
    return wrapped_get_features


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
    chunk_sample_number = int(((chunk_len * sample_number) // d))
    # 'median' o 'edge' (entrambe le direzioni)

    #split matrix of features, chunk starts from 0
    diff = chunk_sample_number - (sample_number % chunk_sample_number)
    x=np.pad(x, ((diff // 2, math.ceil(diff / 2)), (0, 0)), 'edge')
    for chunk in range(0,windows_number):
        start_x = chunk * chunk_sample_number
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
