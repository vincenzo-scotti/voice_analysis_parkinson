# Voice Analysis for Parkinson's Disease

Codebase for the papers "[An Analysis of Features for Machine Learning Approaches to Parkinson's Disease Detection](https://www.overleaf.com/read/dfzwqkspnxxp)" and "[Cross-Lingual Transferability of Voice Analysis Models: a Parkinson's Disease Case Study](https://www.overleaf.com/read/yryxdcgzgprv)" ([extended abstract](https://www.overleaf.com/read/wtjyqvpnngwg)). 
For all the references, contributions and credits, please refer to the paper.

This code was initially developed as part of the M.Sc. Thesis in Computer Science and Engineering "[Cross-Lingual Transferability of Voice Analysis Models: a Parkinson's Disease Case Study](https://www.overleaf.com/read/tkbjcxxjrzjb)" ([executive summary](https://www.overleaf.com/read/qdszgphmhcsd)).
The M.Sc. degree was released by the Dipartimento di Elettronica, Informazione e Bioingengeria  ([DEIB](https://www.deib.polimi.it/eng/home-page)) of the Politecnico di Milano University ([PoliMI](https://www.unitn.it)).
The Thesis was supervised at PoliMI by the staff of the [ARCSlab](https://arcslab.dei.polimi.it).

## Repository structure

This repository is organised into four main directories:

- `experiments/` contains the directories to host:  
    - results of the experiments;
    - experiment configuration dumps.
- `notebooks/` contains the Jupyter notebooks to do:  
    - data exploration;
    - results analysis.
- `resources/` contains:
    - directories to host the dialogue corpora used in the experiments, and the references to download them;
    - directory to host the YAML configuration files to run the experiments.
- `src/` contains modules and scripts to: 
    - fit and evaluate models;
    - extract audio features;
    - preprocess data.

For further details, refer to the `README.md` within each directory.

## Installation

To install all the required packages, instead, run the following commands:

```bash
# Create anaconda environment
conda create -n prkns python=3.10
# Activate anaconda environment
conda activate prkns
# Install packages
pip install -r requirements.txt
```

Finally, download and initialise the RNNoise submodule

```bash
# Install RNNoise submodule 
git submodule init; git submodule update
```

Once the RNNoise submodule is initialised, follow the instructions in `rnnoise/README.md` to install it.

To add the source code directory to the Python path, you can add this line to the file `~/.bashrc`

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/voice_analysis_parkinson/src
```

Finally, make sure that [FFMpeg](https://ffmpeg.org) is installed and available on `$PATH`.

## Data preprocessing

There are two scripts for data preprocessing in `./src/bin/utils/`:
1) `./src/bin/utils/preprocess_data.py` can be used to denoise the file within a directory.
2) `./src/bin/utils/prepare_data.py` can be used to apply segments splitting (given the split timings).

## Run experiments

There is a script to run the experiments, it expects to have `./src` in the Python path and all data sets to be downloaded and placed in the `./resources/data/raw/` directory.

To run the script in foreground:
```bash
python ./src/bin/main.py --configs_file_path ./resources/configs/path/to/config.yaml
```

To run the script in background:

```bash
nohup python ./src/bin/main.py --configs_file_path ./resources/configs/path/to/config.yaml > experiment_"$(date '+%Y_%m_%d_%H_%M_%S')".out &
``` 

## References

If you are willing to use our code or our models, please cite our work through the following BibTeX entries:

```bibtex

```
```bibtex
@inbook{ferrante-etal-2023-analysis,
	address = {Boca Raton},
	author = {Ferrante, Claudio and Menon, Bindu and Pillai, Anitha S. and Sbattella, Licia and Scotti, Vincenzo},
	booktitle = {Machine Learning and Deep Learning in Natural Language Processing},
	doi = {10.1201/9781003296126-13},
	editor = {Pillai, Anitha S. and Tedesco, Roberto},
	isbn = {978-1-003-29612-6},
	pages = {169--183},
	publisher = {CRC Press},
	title = {An Analysis of Features for Machine Learning Approaches to Parkinson's Disease Detection},
	url = {https://doi.org/10.1201/9781003296126-13},
	year = {2023},
}
```
```bibtex
@conference{ferrante-scotti-2023-cross,
	author = {Ferrante, Claudio and Scotti, Vincenzo},
	booktitle = {Booklet of abstracts -- SPOKEN LANGUAGE IN THE MEDICAL FIELD: Linguistic analysis, technological applications and clinical tools},
	pages = {40--42},
	title = {Cross-lingual transferability of voice analysis models: a Parkinson's Disease case study},
	url = {https://www.aisv.it/lecce2023/BookOfAbstract_Lecce2023.pdf#page=40},
	year = {2023}
}
```
