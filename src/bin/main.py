"""
This is the main script.
The script takes the path to the source directory of the English data set, and that of the Hindi data set.
The directories will contain cleaned and cut audio clips to classify

For each of the considered features the script will train and test a model on the English corpus
and then evaluate that same model on the Hindi corpus.
All evaluation results will be logged to make comparisons between the models results.
"""
import os
import sys
from shutil import copy2
from datetime import datetime
from argparse import ArgumentParser, Namespace

import librosa
import yaml

from typing import List, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from features import GlobalPooling, pooling, FEATURE_EXTRACTORS, trunc_audio


def main(args: Namespace):
    # Prepare environment (load configs)
    # Get date-time
    date_time_experiment: str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    # Read YAML file with configurations
    with open(args.configs_file_path) as f:
        configs: Dict = yaml.full_load(f)
    # Create experiment directory
    current_experiment_dir_path = os.path.join(args.experiments_dir_path, 'experiment_' + date_time_experiment)
    os.mkdir(current_experiment_dir_path)
    # Dump configs
    copy2(args.configs_file_path, os.path.join(current_experiment_dir_path, 'configs.yaml'))
    # Output file path
    output_file_path = os.path.join(current_experiment_dir_path, 'results.txt')
    # Label encoder
    le = LabelEncoder()

    # Load source language audio paths and labels
    src_language_metadata_df = pd.read_csv(os.path.join(args.source_language_data_dir_path, 'metadata.csv'))
    X_src_paths: List[str] = [
        os.path.join(args.source_language_data_dir_path, file_name) for file_name in src_language_metadata_df.file_name
    ]
    X_src_idxs: np.ndarray = np.arange(len(X_src_paths))
    y_src_labels: np.ndarray = le.fit_transform([label for label in src_language_metadata_df.label])
    # Generate train-test split
    X_src_train_idxs, X_src_test_idxs, y_src_train, y_src_test = train_test_split(
        X_src_idxs, y_src_labels, random_state=configs.get('random_seed', None)
    )

    # Load target language audio paths and labels
    tgt_language_metadata_df = pd.read_csv(os.path.join(args.target_language_data_dir_path, 'metadata.csv'))
    X_tgt_paths: List[str] = [
        os.path.join(args.target_language_data_dir_path, file_name) for file_name in tgt_language_metadata_df.file_name
    ]
    y_tgt_labels: np.ndarray = le.transform([label for label in tgt_language_metadata_df.label])

    # Compute audio file durations in seconds
    X_src_duration: List[float] = [librosa.get_duration(filename=path) for path in X_src_paths]
    X_tgt_duration: List[float] = [librosa.get_duration(filename=path) for path in X_tgt_paths]

    # For each feature
    for feature in FEATURE_EXTRACTORS:
        # Extract feature sequences
        X_src: List[np.ndarray] = [
            FEATURE_EXTRACTORS[feature](
                path,
                *configs['features'][feature].get('args', tuple()),
                **configs['features'][feature].get('kwargs', dict())
            )
            for path in X_src_paths
        ]
        X_tgt: List[np.ndarray] = [
            FEATURE_EXTRACTORS[feature](
                path,
                *configs['features'][feature].get('args', tuple()),
                **configs['features'][feature].get('kwargs', dict())
            )
            for path in X_tgt_paths
        ]

        # Apply train-test splitting to source data (including durations)
        X_src_train: List[np.ndarray] = [X_src[i] for i in X_src_train_idxs]
        X_src_train_duration: List[float] = [X_src[i] for i in X_src_train_idxs]
        X_src_test: List[np.ndarray] = [X_src[i] for i in X_src_test_idxs]
        X_src_test_duration: List[float] = [X_src[i] for i in X_src_test_idxs]

        # Apply chunking to all feature maps and and train-test splitting to source data
        X_src_train, y_src_train = list(zip(*sum([
            trunc_audio(X, y, d, chunk_len=configs.get('chunk_duration', 4.0))
            for X, y, d in zip(X_src_train, y_src_train, X_src_train_duration)
        ], [])))
        X_src_test, y_src_test = list(zip(*sum([
            trunc_audio(X, y, d, chunk_len=configs.get('chunk_duration', 4.0))
            for X, y, d in zip(X_src_test, y_src_test, X_src_test_duration)
        ], [])))
        X_tgt, y_tgt_labels = list(zip(*sum([
            trunc_audio(X, y, d, chunk_len=configs.get('chunk_duration', 4.0))
            for X, y, d in zip(X_tgt, y_tgt_labels, X_tgt_duration)
        ], [])))
        # TODO call utils function to split audio further, (also keep track of label)
        # splith both y_source (english) and y_target (hindi)

        # For each pooling approach
        for t_pooling in GlobalPooling:
            # Get source language features
            X_src_train: np.ndarray = np.vstack([pooling(x, t_pooling)] for x in X_src_train)
            X_src_test: np.ndarray = np.vstack([pooling(x, t_pooling)] for x in X_src_test)
            # Do standardisation
            std_scaler: StandardScaler = StandardScaler().fit(X_src_train)
            X_src_train: np.ndarray = std_scaler.transform(X_src_train)
            X_src_test: np.ndarray = std_scaler.transform(X_src_test)
            # Do PCA
            pca: PCA = PCA(n_components=configs.get('pca_components', 0.9)).fit(X_src_train)
            X_src_train: np.ndarray = pca.transform(X_src_train)
            X_src_test: np.ndarray = pca.transform(X_src_test)

            # Get target language features
            X_tgt: np.ndarray = np.vstack([pooling(x, t_pooling)] for x in X_tgt)
            # Do standardisation (using already fit model from source language)
            X_tgt: np.ndarray = std_scaler.transform(X_tgt)
            # Do PCA (using already fit model from source language)
            X_tgt: np.ndarray = pca.transform(X_tgt)

            # TODO add domain adaptation

            # Train classifier (on source data)
            cls: SVC = SVC()
            cls.fit(X_src_train, y_src_train)
            # Test classifier (on source data)
            y_src_test_pred = cls.predict(X_src_test)
            y_src_test_proba = cls.predict_proba(X_src_test)
            src_cls_report = classification_report(y_src_test, y_src_test_pred)
            src_auc_score = roc_auc_score(y_src_test, y_src_test_proba)
            src_confusion_matrix = confusion_matrix(y_src_test, y_src_test_pred)

            # Test classifier (on target data)
            y_tgt_pred = cls.predict(X_tgt)
            y_tgt_proba = cls.predict(X_tgt)
            tgt_cls_report = classification_report(y_tgt_labels, y_tgt_pred)
            tgt_auc_score = roc_auc_score(y_tgt_labels, y_tgt_proba)
            tgt_confusion_matrix = confusion_matrix(y_tgt_labels, y_tgt_pred)
            
            # TODO do calibration? 
            # Create a notebook for visualisation of results?

            # Log results
            with open(output_file_path, 'a') as f:
                print(f"Feature type: {feature}, pooling type: {t_pooling}\n", file=f)
                print("Source language classification results\n", file=f)
                print(f"Classification report: \n{src_cls_report}\n", file=f)
                print(f"ROC AUC score: {src_auc_score}\n", file=f)
                print(f"Confusion matrix (true label on row, predicted on columns): \n{src_confusion_matrix}\n", file=f)
                print("\n\n")
                print("Target language classification results\n", file=f)
                print(f"Classification report: \n{tgt_cls_report}\n", file=f)
                print(f"ROC AUC score: {tgt_auc_score}\n", file=f)
                print(f"Confusion matrix (true label on row, predicted on columns): \n{tgt_confusion_matrix}\n", file=f)
                print("\n\n\n\n")

    return 0


if __name__ == "__main__":
    # Instantiate argument parser
    args_parser: ArgumentParser = ArgumentParser()
    # Add arguments to parser
    args_parser.add_argument(
        '--configs_file_path', type=str, default='./resources/configs/configs.yaml',
        help="Path to the YAML file with the configuration."
    )
    args_parser.add_argument(
        '--source_language_data_dir_path', type=str, default='./resources/data/english_corpus/',
        help="Path to the directory with the audio clips in the source language for the experiments."
    )
    args_parser.add_argument(
        '--target_language_data_dir_path', type=str, default='./resources/data/hindi_corpus/',
        help="Path to the directory with the audio clips in the source language for the experiments."
    )
    args_parser.add_argument(
        '--experiments_dir_path', type=str, default='./experiments/',
        help="Path to the directory to store the results of the experiments."
    )
    # Run experiment
    main(args_parser.parse_args(sys.argv[1:]))
