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

from typing import List, Dict, Union, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from adapt.feature_based import CORAL

from features import GlobalPooling, pooling, FEATURE_EXTRACTORS, trunc_audio


def main(args: Namespace):
    # Prepare environment (load configs)
    # Get date-time
    date_time_experiment: str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    # Read YAML file with configurations
    with open(args.configs_file_path) as f:
        configs: Dict = yaml.full_load(f)
    # Create experiment directory
    current_experiment_dir_path = os.path.join(configs['experiments_dir'], 'experiment_' + date_time_experiment)
    os.mkdir(current_experiment_dir_path)
    # Dump configs
    copy2(args.configs_file_path, os.path.join(current_experiment_dir_path, 'configs.yaml'))
    # Output file path
    output_file_path = os.path.join(current_experiment_dir_path, 'results.txt')
    f = open(output_file_path, "x")
    f.close()
    # Scores data path
    scores_file_path = os.path.join(current_experiment_dir_path, 'scores.csv')
    # Scores Data Frame columns
    score_cols = ['feature', 'pooling', 'adaptation', 'data_set', 'metric', 'value']
    # Accumulator for scores data
    score_data: List[Tuple[str, str, bool, str, str, Union[float, List[float], int, List[List[int]]]]] = []
    # Label encoder
    le = LabelEncoder()

    # Load source language audio paths and labels
    src_language_metadata_df = pd.read_csv(os.path.join(configs['src_lang_dir'], 'metadata.csv'))
    X_src_paths: List[str] = [
        os.path.join(configs['src_lang_dir'], file_name) for file_name in src_language_metadata_df.file_name
        if os.path.exists(os.path.join(configs['src_lang_dir'], file_name))
    ]
    X_src_idxs: np.ndarray = np.arange(len(X_src_paths))
    y_src_labels: np.ndarray = le.fit_transform([label for label in src_language_metadata_df.label])
    # Generate train-test split
    X_src_train_idxs, X_src_test_idxs, y_src_train, y_src_test = train_test_split(
        X_src_idxs, y_src_labels, random_state=configs.get('random_seed', None)
    )

    # Load target language audio paths and labels
    tgt_language_metadata_df = pd.read_csv(os.path.join(configs['tgt_lang_dir'], 'metadata.csv'))
    X_tgt_paths: List[str] = [
        os.path.join(configs['tgt_lang_dir'], file_name) for file_name in tgt_language_metadata_df.file_name
        if os.path.exists(os.path.join(configs['tgt_lang_dir'], file_name))
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
                path, cache_dir_path=configs.get('cache_path')
            )
            for path in X_src_paths
        ]
        X_tgt: List[np.ndarray] = [
            FEATURE_EXTRACTORS[feature](
                path, cache_dir_path=configs.get('cache_path')
            )
            for path in X_tgt_paths
        ]

        # Apply train-test splitting to source data (including durations)
        X_src_train: List[np.ndarray] = [X_src[i] for i in X_src_train_idxs]
        X_src_train_duration: List[float] = [X_src_duration[i] for i in X_src_train_idxs]
        X_src_test: List[np.ndarray] = [X_src[i] for i in X_src_test_idxs]
        X_src_test_duration: List[float] = [X_src_duration[i] for i in X_src_test_idxs]

        # TODO find better solution for corrupted data management
        X_src_train_tmp, y_src_train_tmp, X_src_train_duration_tmp = [*zip(*[
            (x, y, d) for x, y, d in zip(X_src_train, y_src_train, X_src_train_duration) if x is not None
        ])]
        X_src_test_tmp, y_src_test_tmp, X_src_test_duration_tmp = [*zip(*[
            (x, y, d) for x, y, d in zip(X_src_test, y_src_test, X_src_test_duration) if x is not None
        ])]
        X_tgt_tmp, y_tgt_labels_tmp, X_tgt_duration_tmp = [*zip(*[
            (x, y, d) for x, y, d in zip(X_tgt, y_tgt_labels, X_tgt_duration) if x is not None
        ])]

        # Apply chunking to all feature maps and and train-test splitting to source data
        X_src_train, y_src_train_split = list(zip(*sum([
            trunc_audio(X, y, d, chunk_len=configs.get('chunk_duration', 4.0))
            for X, y, d in zip(X_src_train_tmp, y_src_train_tmp, X_src_train_duration_tmp)
        ], [])))
        X_src_test, y_src_test_split = list(zip(*sum([
            trunc_audio(X, y, d, chunk_len=configs.get('chunk_duration', 4.0))
            for X, y, d in zip(X_src_test_tmp, y_src_test_tmp, X_src_test_duration_tmp)
        ], [])))
        X_tgt, y_tgt_labels_split = list(zip(*sum([
            trunc_audio(X, y, d, chunk_len=configs.get('chunk_duration', 4.0))
            for X, y, d in zip(X_tgt_tmp, y_tgt_labels_tmp, X_tgt_duration_tmp)
        ], [])))

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

            for do_adaptation in [False, True]:
                # Train classifier (on source data)
                cls: SVC = SVC(probability=True)
                if do_adaptation:
                    adapter = CORAL(random_state=configs.get('random_seed', None))
                    X_src_train_adapted = adapter.fit_transform(X_src_train, X_tgt)
                    X_src_test_adapted = adapter.transform(X_src_test)
                else:
                    X_src_train_adapted = X_src_train
                    X_src_test_adapted = X_src_test
                cls.fit(X_src_train_adapted, y_src_train_split)
                # Test classifier (on source data)
                y_src_test_pred = cls.predict(X_src_test_adapted)
                y_src_test_proba = cls.predict_proba(X_src_test_adapted)[:, 1]

                src_cls_report = classification_report(y_src_test_split, y_src_test_pred)
                src_accuracy_score = accuracy_score(y_src_test_split, y_src_test_pred)
                src_precision_score, src_recall_score, src_fscore, src_support = precision_recall_fscore_support(
                    y_src_test_split, y_src_test_pred, average='macro'
                )
                src_specificity_score = recall_score(y_src_test_split, y_src_test_pred, average='macro', pos_label=0)
                src_auc_score = roc_auc_score(y_src_test_split, y_src_test_proba)
                src_fpr, src_tpr, src_roc_threshold = roc_curve(y_src_test_split, y_src_test_proba)
                src_pr, src_rc, src_pr_rc_threshold = precision_recall_curve(y_src_test_split, y_src_test_proba)
                src_confusion_matrix = confusion_matrix(y_src_test_split, y_src_test_pred)

                # Test classifier (on target data)
                y_tgt_pred = cls.predict(X_tgt)
                y_tgt_proba = cls.predict_proba(X_tgt)[:, 1]

                tgt_cls_report = classification_report(y_tgt_labels_split, y_tgt_pred)
                tgt_accuracy_score = accuracy_score(y_tgt_labels_split, y_tgt_pred)
                tgt_precision_score, tgt_recall_score, tgt_fscore, tgt_support = precision_recall_fscore_support(
                    y_tgt_labels_split, y_tgt_pred, average='macro'
                )
                tgt_specificity_score = recall_score(y_tgt_labels_split, y_tgt_pred, average='macro', pos_label=0)
                tgt_auc_score = roc_auc_score(y_tgt_labels_split, y_tgt_proba)
                tgt_fpr, tgt_tpr, tgt_roc_threshold = roc_curve(y_tgt_labels_split, y_tgt_proba)
                tgt_pr, tgt_rc, tgt_pr_rc_threshold = precision_recall_curve(y_tgt_labels_split, y_tgt_proba)

                tgt_confusion_matrix = confusion_matrix(y_tgt_labels_split, y_tgt_pred)

                # Log results
                with open(output_file_path, 'a') as f:
                    print(f"Feature type: {feature}, pooling type: {t_pooling.value}, domain adaptation: {do_adaptation}\n", file=f)
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
                # Append scores
                score_data += [
                    (feature, t_pooling.value, do_adaptation, 'src', 'accuracy', src_accuracy_score),
                    (feature, t_pooling.value, do_adaptation, 'src', 'precision', src_precision_score),
                    (feature, t_pooling.value, do_adaptation, 'src', 'recall', src_recall_score),
                    (feature, t_pooling.value, do_adaptation, 'src', 'fscore', src_fscore),
                    (feature, t_pooling.value, do_adaptation, 'src', 'support', src_support),
                    (feature, t_pooling.value, do_adaptation, 'src', 'specificity', src_specificity_score),
                    (feature, t_pooling.value, do_adaptation, 'src', 'roc_auc', src_auc_score),
                    (feature, t_pooling.value, do_adaptation, 'src', 'fpr', src_fpr.tolist()),
                    (feature, t_pooling.value, do_adaptation, 'src', 'tpr', src_tpr.tolist()),
                    (feature, t_pooling.value, do_adaptation, 'src', 'roc_thresholds', src_roc_threshold.tolist()),
                    (feature, t_pooling.value, do_adaptation, 'src', 'precisions', src_pr.tolist()),
                    (feature, t_pooling.value, do_adaptation, 'src', 'recalls', src_rc.tolist()),
                    (feature, t_pooling.value, do_adaptation, 'src', 'pr_rc_thresholds', src_pr_rc_threshold.tolist()),
                    (feature, t_pooling.value, do_adaptation, 'src', 'confusion_matrix', src_confusion_matrix.tolist()),
                    (feature, t_pooling.value, do_adaptation, 'tgt', 'accuracy', tgt_accuracy_score),
                    (feature, t_pooling.value, do_adaptation, 'tgt', 'precision', tgt_precision_score),
                    (feature, t_pooling.value, do_adaptation, 'tgt', 'recall', tgt_recall_score),
                    (feature, t_pooling.value, do_adaptation, 'tgt', 'fscore', tgt_fscore),
                    (feature, t_pooling.value, do_adaptation, 'tgt', 'support', tgt_support),
                    (feature, t_pooling.value, do_adaptation, 'tgt', 'specificity', tgt_specificity_score),
                    (feature, t_pooling.value, do_adaptation, 'tgt', 'roc_auc', tgt_auc_score),
                    (feature, t_pooling.value, do_adaptation, 'tgt', 'fpr', tgt_fpr.tolist()),
                    (feature, t_pooling.value, do_adaptation, 'tgt', 'tpr', tgt_tpr.tolist()),
                    (feature, t_pooling.value, do_adaptation, 'tgt', 'roc_thresholds', tgt_roc_threshold.tolist()),
                    (feature, t_pooling.value, do_adaptation, 'tgt', 'precisions', tgt_pr.tolist()),
                    (feature, t_pooling.value, do_adaptation, 'tgt', 'recalls', tgt_rc.tolist()),
                    (feature, t_pooling.value, do_adaptation, 'tgt', 'pr_rc_thresholds', tgt_pr_rc_threshold.tolist()),
                    (feature, t_pooling.value, do_adaptation, 'tgt', 'confusion_matrix', tgt_confusion_matrix.tolist())
                ]
    # Create data frame with measured scores
    df: pd.DataFrame = pd.DataFrame(score_data, columns=score_cols)
    # Save data frame
    df.to_csv(scores_file_path, index=False)

    return 0


if __name__ == "__main__":
    # Disable CUDA devices (use only CPU)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # Instantiate argument parser
    args_parser: ArgumentParser = ArgumentParser()
    # Add arguments to parser
    args_parser.add_argument(
        '--configs_file_path', type=str, default='./resources/configs/configs.yaml',
        help="Path to the YAML file with the configuration."
    )
    # Run experiment
    main(args_parser.parse_args(sys.argv[1:]))
