# Voice Analysis for Parkinson's Disease

Codebase for the paper "[Cross-Lingual Transferability of Voice Analysis Models: a Parkinson's Disease Case Study]()". 
For all the references, contributions and credits, please refer to the paper.

This code was initially developed as part of the M.Sc. Thesis in Computer Science and Engineering "[Cross-Lingual Transferability of Voice Analysis Models: a Parkinson's Disease Case Study]()".
The M.Sc. degree was released by the Dipartimento di Elettronica, Informazione e Bioingengeria  ([DEIB](https://www.deib.polimi.it/eng/home-page)) of the Politecnico di Milano University ([PoliMI](https://www.unitn.it)).
The Thesis was supervised at PoliMI by the staff of the [ARCSlab](https://arcslab.dei.polimi.it).

## Repository structure

This repository is organised into four main directories:

- `experiments/` contains the directories to host:  
    - results of the experiments 
    - checkpoints generated during the experiments;
    - experiment configuration dumps;
    - experiment logs.
- `resources/` contains:
    - directories to host the dialogue corpora used in the experiments, and the references to download them;
    - directory to host the YAML configuration files to run the experiments.
    - directory to host the pre-trained models, and the references to download them.
- `src/` contains modules and scripts to: 
    - run training steps;
    - run evaluation steps;
    - interact with the trained models;
    - preprocess corpora.

For further details, refer to the `README.md` within each directory.

## Installation

To install all the required packages, instead, run the following commands:

```bash
# Create anaconda environment (skip cudatoolkit option if you don't want to use the GPU)
conda create -n parkison python=3.10 cudatoolkit=11.3
# Activate anaconda environment
conda activate parkison
# Install packages
```

## References

If you are willing to use our code or our models, please cite our work through the following BibTeX entry:

```bibtex

```