from .soundnet import get_soundnet_features
from .vggish import get_vggish_features
from .spectral import get_spectral_features
from .wav2vec import get_wav2vec_features
from .utils import GlobalPooling, pooling, get_audio_length, trunc_audio

from typing import Dict

FEATURE_EXTRACTORS: Dict = {
    'spectral': get_spectral_features,
    'vggish': get_vggish_features,
    'soundnet': get_soundnet_features,
    'wav2vec': get_wav2vec_features
}
