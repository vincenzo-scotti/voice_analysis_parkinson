from typing import Optional

import librosa
import numpy as np
from src.features.utils import pooling, GlobalPooling, trunc_audio
import torch
import os

vggish: Optional = None


def get_vggish():
    global vggish
    if vggish is None:
        vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
        vggish.eval()
    return vggish


def get_vggish_features(
        file_path: str,
        t_pooling: Optional[GlobalPooling] = None,
) -> np.ndarray:
    model = get_vggish()

    #processing data
    tensor_audio_features = model.forward(file_path)
    audio_features = tensor_audio_features.detach().cpu().numpy()
    if t_pooling is not None:
        return pooling(audio_features, t_pooling)
    else:
        return audio_features


#FOR DEBUG PURPOSE
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    a = get_vggish()
    a.eval()
    res = a.forward("resources/data/Split_denoised_hindi/AUD-20210515-WA0002_000.wav")
    #print(type(res.detach().cpu().numpy()))

    #print(res.detach().cpu().numpy())
    #TEST TRUNC_AUDIO
    y=  np.asarray([1])
    duration = librosa.get_duration(filename="resources/data/Split_denoised_hindi/AUD-20210515-WA0002_000.wav")
    prova = trunc_audio(res.detach().cpu().numpy(),y,duration,4.0)
    print(type(prova),len(prova),prova)
