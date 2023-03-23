# Data

This directory is used to host the data set(s).
Data set(s) are available at the following links:

- English:
    - [MDVR-KCL](https://zenodo.org/record/2867216#.YpSK3i0QPT9): Dataset of Mobile Device Voice Recordings at King's College London from both early and advanced Parkinson's disease patients hosted by [Zenodo](https://zenodo.org)
- Telugu:
    - [Parkison corpus in Telugu](https://drive.google.com/drive/folders/1lBz_NhtP0o0uy3OMV6FAtTHkaA4Prmf-?usp=share_link): Private collection of voice recordings of Parkinson's disease patients in Hindi hosted by [Prof. Anitha S.Pillai](https://www.linkedin.com/in/anithaspillai/)
    - [OpenSLR Telugu split](https://www.aclweb.org/anthology/2020.lrec-1.800): Voice recordings in Telugu of people uttering sentences for ASR development hosted by [OpenSLR](https://www.openslr.org/66/)
- Italian:
    - [Italian Parkinson's Voice and Speech](https://ieee-dataport.org/open-access/italian-parkinsons-voice-and-speech): Voice recordings of Italian Parkinson's disease patients and healthy individuals, hosted by [IEEE Data Port](https://ieee-dataport.org/)

Notes: 
- Audio clips undergo denoising through the [RNNoise](https://jmvalin.ca/demo/rnnoise/) tool from [Mozilla](https://www.mozilla.org/). The tool is available open source on GitLab at the following [link](https://gitlab.xiph.org/xiph/rnnoise/), follow the README instructions for installation and demo usage.
- Samples from Common Voice were converted from MP3 to WAV format manually. They were adopted to balance the Hindi corpus and have also examples of healthy individuals. Metadata about the samples were directly added to the 
- The Italian data set is put only for reference for now, we did not use it during the experiments.

Directory structure:
```
 |- data/
    |- preprocessed/
      |- en/
        |- metadata.csv (for training script)
        |- ...
      |- hi/
        |- metadata.csv (for training script)
        |- ...
    |- raw/
      |- en/
        |- 26-29_09_2017_KCL/
          |- metadata.csv (for preprocessing script)
          |- ... (original directory structure from MDVR-KCL)
      |- te/
        |- asp/
          |- Parkinson's voice/
            |- metadata.csv (for denoising script)
            |- ... (raw clips)
          |- Parkinson's voice - denoised/
            |- metadata.csv (for preprocessing script)
            |- ... (denoised clips)
        |- OpenSLR/
          |- metadata.csv (subsampled randomly)
          |- te_in_female/
            |- LICENSE
            |- line_index.tsv
            |- ...
          |- te_in_male/
            |- LICENSE
            |- line_index.tsv
            |- ...
```