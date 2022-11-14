from typing import Optional
import numpy as np
from src.features.utils import load_audio,pooling, GlobalPooling
import torch, torchaudio
import torch.nn as nn
torchaudio.set_audio_backend("soundfile")
soundnet: Optional = None

#Soundnet model class


class SoundNet8_pytorch(nn.Module):
    def __init__(self):
        super(SoundNet8_pytorch, self).__init__()

        self.define_module()

    def define_module(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, (64, 1), (2, 1), (32, 0), bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((8, 1), (8, 1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (32, 1), (2, 1), (16, 0), bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((8, 1), (8, 1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, (16, 1), (2, 1), (8, 0), bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, (8, 1), (2, 1), (4, 0), bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, (4, 1), (2, 1), (2, 0), bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((4, 1), (4, 1))
        )  # difference here (0.24751323, 0.2474), padding error has beed debuged
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, (4, 1), (2, 1), (2, 0), bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 1024, (4, 1), (2, 1), (2, 0), bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(1024, 1000, (8, 1), (2, 1), (0, 0), bias=True),
        )
        self.conv8_2 = nn.Sequential(
            nn.Conv2d(1024, 401, (8, 1), (2, 1), (0, 0), bias=True)
        )

    def forward(self, x):
        for net in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7]:
            x = net(x)
        object_pred = self.conv8(x)
        scene_pred = self.conv8_2(x)
        return object_pred, scene_pred

    def extract_feat(self, x: torch.Tensor) -> list:
        output_list = []
        for net in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7]:
            x = net(x)
            output_list.append(x.detach().cpu().numpy())
        # object_pred = self.conv8(x)
        # output_list.append(object_pred.detach().cpu().numpy())
        # scene_pred = self.conv8_2(x)
        # output_list.append(scene_pred.detach().cpu().numpy())
        return output_list





def get_soundnet():
    global soundnet
    if soundnet is None:
        soundnet = SoundNet8_pytorch()
        soundnet.load_state_dict(torch.load("./sound8.pth"))

    return soundnet


def get_soundnet_features(
        file_path: str,
        *model_args,
        t_pooling: Optional[GlobalPooling] = None,
        tgt_len: Optional[float] = None,
        **model_kwargs
) -> np.ndarray:
    raw_data, sample_rate = torchaudio.load(file_path)
    model = get_soundnet()
    resampler = torchaudio.transforms.Resample(48000, 22050)
    raw_data = resampler.forward(raw_data)
    raw_data = raw_data.unsqueeze(1).unsqueeze(-1)

    #Processing data
    feats = model.extract_feat(raw_data)
    audio_features = feats[6].squeeze(0).squeeze(-1).T
    #output shape (7,1024)
    if pooling is not None:
        return pooling(audio_features, t_pooling)
    else:
        return audio_features


# #FOR DEBUG PURPOSE
# if __name__ == "__main__":
#     a = get_soundnet()
#     # metadata = torchaudio.info("resources/data/Split_denoised_hindi/AUD-20210515-WA0002_000.wav")
#     # print(metadata)
#     raw_data, sample_rate = torchaudio.load("resources/data/Split_denoised_hindi/AUD-20210515-WA0002_000.wav")
#     resampler = torchaudio.transforms.Resample(48000,22050)
#     print(raw_data.shape)
#     raw_data=resampler.forward(raw_data)
#     print(raw_data.shape)
#     raw_data = raw_data.unsqueeze(1).unsqueeze(-1)
#     print(raw_data.shape)
#     feats = a.extract_feat(raw_data)
#     # features for layer1 to layer8
#     print(feats[6].squeeze(0).squeeze(-1).T.shape)
#     # for idx, f in enumerate(feats):
#     #     f= f.squeeze().T
#     #     print(f"feature shape for layer {idx}: {f.shape}")
