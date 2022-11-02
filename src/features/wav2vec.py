from typing import Optional
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model,Wav2Vec2CTCTokenizer, AutoFeatureExtractor
from .utils import load_audio, pooling, GlobalPooling

wav2vec: Optional = None
processorWav2vec: Optional = None

def get_wav2vec():
    global wav2vec
    global processorWav2vec
    if wav2vec is None:
        wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    if processorWav2vec is None:
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
        processorWav2vec = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    return wav2vec, processorWav2vec

def get_wav2vec_features(
        file_path: str,
        *model_args,
        t_pooling: Optional[GlobalPooling] = None,
        tgt_len: Optional[float] = None,
        **model_kwargs
    ) -> np.ndarray:

    raw_data, sample_rate = load_audio(file_path, tgt_len=tgt_len)
    model,processor = get_wav2vec()

    #processing data
    inputs = processor(raw_data, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    audio_features = torch.squeeze(outputs.last_hidden_state).numpy()
    #print(list(audio_features.shape))

    if pooling is not None:
        return pooling(audio_features, t_pooling)
    else:
        return audio_features
