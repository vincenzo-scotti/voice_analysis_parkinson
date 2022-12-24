from typing import Optional
import numpy as np
from src.features.utils import pooling, GlobalPooling, trunc_audio
import torch
from .utils import FeaturesCache, safe_features_load

vggish: Optional = None


def get_vggish():
    global vggish
    if vggish is None:
        vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
        vggish.eval()
    return vggish


@safe_features_load
def get_vggish_features(
        file_path: str,
        t_pooling: Optional[GlobalPooling] = None,
        cache_dir_path: Optional[str] = None
) -> np.ndarray:
    # Look in cache
    if cache_dir_path is not None:
        cache = FeaturesCache(cache_dir_path, 'vggish')
        audio_features = cache.get_cached_features(file_path)
    else:
        cache = audio_features = None
    # If not in cache load
    if audio_features is None:
        model = get_vggish()
        #processing data
        tensor_audio_features = model.forward(file_path)
        audio_features = tensor_audio_features.detach().cpu().numpy()
        if len(audio_features.shape) == 1:
            audio_features = audio_features.reshape(1, -1)
        # Cache data if needed
        if cache is not None:
            cache.cache_features(file_path, audio_features)
    # Apply pooling (if required)
    if t_pooling is not None:
        return pooling(audio_features, t_pooling)
    else:
        return audio_features


# #FOR DEBUG PURPOSE
# if __name__ == "__main__":
#     os.environ["CUDA_VISIBLE_DEVICES"] = ""
#     a = get_vggish()
#     a.eval()
#     res = a.forward("resources/data/Split_denoised_hindi/AUD-20210515-WA0002_000.wav")
#     #print(type(res.detach().cpu().numpy()))
#
#     #print(res.detach().cpu().numpy())
#     #TEST TRUNC_AUDIO
#     y=  np.asarray([1])
#     duration = librosa.get_duration(filename="resources/data/Split_denoised_hindi/AUD-20210515-WA0002_000.wav")
#     prova = trunc_audio(res.detach().cpu().numpy(),y,duration,4.0)
#     print(type(prova),len(prova),prova)
