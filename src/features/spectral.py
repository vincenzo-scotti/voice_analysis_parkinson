from typing import Optional
from src.features.utils import load_audio,pooling, GlobalPooling
import librosa
import parselmouth
from parselmouth.praat import call
import numpy as np
import math
from joblib import Parallel
from joblib import delayed
from joblib import parallel_backend
from .utils import FeaturesCache, safe_features_load


#GLOBAL PARAMETERS
SAMPLE_RATE = 16000
WINDOW_LENGHT = 1024
HOP_LENGHT = 256
WINDOW_LENGHT_SEC = WINDOW_LENGHT/SAMPLE_RATE
HOP_LENGHT_SEC = HOP_LENGHT/SAMPLE_RATE #
F0MIN = 75
F0MAX = 300

# output (tempo,feature) e.g. (586,13)
def get_mfcc(raw_data,sample_rate):
    mfcc_features_matrix = librosa.feature.mfcc(y=raw_data, sr=sample_rate, n_mfcc = 13,n_fft=WINDOW_LENGHT, hop_length=HOP_LENGHT)
    return mfcc_features_matrix.T



def get_features(t_in,duration,pitch,sound,f0min,f0max):
    t_fin = min(t_in + WINDOW_LENGHT_SEC, duration)
    unit = "Hertz"
    v1 = []
    # TODO fare una funzione che ti crea v2 con joblib
    v1.append(call(pitch, "Get mean", t_in, t_fin, unit))  # get mean pitch (mean F0)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    v1.append(call(harmonicity, "Get mean", t_in, t_fin))  # hnr
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    v1.append(call(pointProcess, "Get jitter (local)", t_in, t_fin, 0.0001, 0.02, 1.3))  # local jitter
    v1.append(call(pointProcess, "Get jitter (local, absolute)", t_in, t_fin, 0.0001, 0.02, 1.3))  # localAbsoluteJitter
    v1.append(call(pointProcess, "Get jitter (rap)", t_in, t_fin, 0.0001, 0.02, 1.3))  # rapjitter
    v1.append(call(pointProcess, "Get jitter (ppq5)", t_in, t_fin, 0.0001, 0.02, 1.3))  # ppq5jitter
    v1.append(call([sound, pointProcess], "Get shimmer (local)", t_in, t_fin, 0.0001, 0.02, 1.3, 1.6))  # localShimmer
    v1.append(
        call([sound, pointProcess], "Get shimmer (local_dB)", t_in, t_fin, 0.0001, 0.02, 1.3, 1.6))  # localdbShimmer
    v1.append(call([sound, pointProcess], "Get shimmer (apq3)", t_in, t_fin, 0.0001, 0.02, 1.3, 1.6))  # apq3Shimmer
    v1.append(call([sound, pointProcess], "Get shimmer (apq5)", t_in, t_fin, 0.0001, 0.02, 1.3, 1.6))  # apq5Shimmer
    return v1

# output (tempo,feature) e.g. (586,11)
def get_acustic_features(file_path,f0min,f0max,duration):
    sound = parselmouth.Sound(file_path)
    pitch = call(sound, "To Pitch", HOP_LENGHT_SEC, f0min, f0max)  # create a praat pitch object
    pitch_vector = np.array([i[0] for i in pitch.selected_array])
    v2=[]
    count=0
    with parallel_backend('threading', n_jobs=-1):
        v2 = Parallel(verbose=2)(delayed(get_features)(t_in,duration,pitch,sound,f0min,f0max) for t_in in np.arange(0,duration,HOP_LENGHT_SEC))
    # print("spectral done")
    v2 = np.vstack(v2)
    lenght_pitch = pitch_vector.shape[0]
    length_v2 = v2.shape[0]
    diff = length_v2 - lenght_pitch
    pitch_vector = np.pad(pitch_vector, (diff//2, math.ceil(diff/2)), 'constant')
    pitch_vector = np.asarray([pitch_vector]).T
    v3 = np.append(pitch_vector, v2, axis=1)
    return v3


@safe_features_load
def get_spectral_features(
        file_path: str,
        t_pooling: Optional[GlobalPooling] = None,
        cache_dir_path: Optional[str] = None
):
    # Look in cache
    if cache_dir_path is not None:
        cache = FeaturesCache(cache_dir_path, 'spectral')
        audio_features = cache.get_cached_features(file_path)
    else:
        cache = audio_features = None
    # If not in cache load
    if audio_features is None:
        data, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
        duration = librosa.get_duration(data, sample_rate)
        # extraction of features
        mfcc_features = get_mfcc(data,sample_rate)
        acustic_features = get_acustic_features(file_path, F0MIN, F0MAX, duration)

        # matrix x axis length validation
        while acustic_features.shape[0] != mfcc_features.shape[0]:
            if acustic_features.shape[0] < mfcc_features.shape[0]:
                mfcc_features = mfcc_features[:-1]
            else:
                acustic_features = acustic_features[:-1]

        # matrix join of columns
        final_features = np.append(mfcc_features, acustic_features, axis=1)
        # print(final_features, final_features.shape)
        # Cache data if needed
        if cache is not None:
            cache.cache_features(file_path, final_features)
    else:
        final_features = audio_features
    # Apply pooling (if required)
    if t_pooling is not None:
        return pooling(final_features, t_pooling)
    else:
        return final_features


# #FOR DEBUG PURPOSE
# if __name__ == "__main__":
#     #print(get_spectral_features(file_path="resources/data/Split_denoised_hindi/AUD-20210515-WA0002_000.wav").shape)
#
#
#     data1, sample_rate = librosa.load("resources/data/Split_denoised_hindi/AUD-20210525-WA0024_000.wav", sr=16000)
#     #GET THE MAX DURATION AS UPPER BOUND
#
#     mfcc = librosa.feature.mfcc(y=data1, sr=16000, n_mfcc = 13,n_fft=WINDOW_LENGHT, hop_length=HOP_LENGHT)
#     mfcc = mfcc.T
#     print(sample_rate,mfcc,mfcc.shape,type(mfcc))
#
#     sound = parselmouth.Sound("resources/data/Split_denoised_hindi/AUD-20210525-WA0024_000.wav")
#     duration = librosa.get_duration(data1,sample_rate)
#     acusticFeatures = get_acustic_features(sound, F0MIN, F0MAX,duration)
#     while acusticFeatures.shape[0] != mfcc.shape[0]:
#         if acusticFeatures.shape[0] < mfcc.shape[0]:
#             mfcc = mfcc[:-1]
#         else:
#             acusticFeatures = acusticFeatures[:-1]
#     finalFeatures = np.append(mfcc, acusticFeatures, axis=1)
#     print(finalFeatures,finalFeatures.shape)






