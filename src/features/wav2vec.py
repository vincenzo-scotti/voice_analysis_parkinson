from typing import Optional
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2CTCTokenizer, AutoFeatureExtractor
from src.features.utils import load_audio, pooling, GlobalPooling
from .utils import FeaturesCache, safe_features_load

wav2vec: Optional = None
processorWav2vec: Optional = None


def get_wav2vec():
    global wav2vec
    global processorWav2vec
    if wav2vec is None:
        wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    if processorWav2vec is None:
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        tokenizer = Wav2Vec2CTCTokenizer("./resources/models/wav2vec/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
        processorWav2vec = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    return wav2vec, processorWav2vec


@safe_features_load
def get_wav2vec_features(
        file_path: str,
        t_pooling: Optional[GlobalPooling] = None,
        tgt_len: Optional[float] = None,
        cache_dir_path: Optional[str] = None
) -> np.ndarray:
    # Look in cache
    if cache_dir_path is not None:
        cache = FeaturesCache(cache_dir_path, 'wav2vec')
        audio_features = cache.get_cached_features(file_path)
    else:
        cache = audio_features = None
    # If not in cache load
    if audio_features is None:
        raw_data, sample_rate = load_audio(file_path, tgt_len=tgt_len, sr=16000)
        model, processor = get_wav2vec()

        # processing data, model pretrained needs sr of 16kHz
        inputs = processor(raw_data, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        audio_features = torch.squeeze(outputs.last_hidden_state).numpy()
        # print(list(audio_features.shape))
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
#     mod,proc = get_wav2vec()
#     raw_data, sample_rate = load_audio("resources/data/Split_denoised_hindi/AUD-20210515-WA0002_000.wav", tgt_len=None, sr=16000)
#     inputs = proc(raw_data, sampling_rate=sample_rate, return_tensors="pt")
#     with torch.no_grad():
#         outputs = mod(**inputs)
#     audio_features = torch.squeeze(outputs.last_hidden_state).numpy()
#     print(audio_features,audio_features.shape)

