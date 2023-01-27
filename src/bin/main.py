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

from typing import List, Dict, Union, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
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
from adapt.feature_based import DeepCORAL

from features import GlobalPooling, FEATURE_EXTRACTORS, trunc_audio

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, SeparableConv1D, Dense, Flatten
from tensorflow.keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier


def create_nn(
        input_shape: Optional[Tuple[int]] = None,
        n_classes: Optional[int] = 2,
        hidden_units: Optional[Union[int, List[int]]] = None,
        pooling: Optional[GlobalPooling] = None,
        dropout_rate: float = 0.1
):
    if hidden_units is None:
        hidden_units = []
    elif isinstance(hidden_units, int):
        hidden_units = [hidden_units]

    model = Sequential()
    if input_shape is not None:
        model.add(Input(shape=input_shape))
    for n_hidden in hidden_units:
        model.add(Dropout(dropout_rate))
        model.add(SeparableConv1D(n_hidden, 3, padding='same', activation='relu'))
    if pooling is not None:
        if pooling == GlobalPooling.AVERAGE:
            model.add(GlobalAveragePooling1D())
        elif pooling == GlobalPooling.MAXIMUM:
            model.add(GlobalMaxPooling1D())
        else:
            model.add(Flatten())
    if n_classes is not None:
        model.add(Dropout(dropout_rate))
        if n_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(n_classes, activation='softmax'))

    return model


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
    src_language_metadata_df = pd.read_csv(os.path.join(configs['src_lang_dir'], 'metadata.csv'))  # .sample(n=10, random_state=2)  # TODO remove sampling
    X_src_paths: List[str]
    X_src_paths, y_src_labels = [*zip(*[
        (os.path.join(configs['src_lang_dir'], file_name), label)
        for file_name, label in zip(src_language_metadata_df.file_name, src_language_metadata_df.label)
        if os.path.exists(os.path.join(configs['src_lang_dir'], file_name))
    ])]
    X_src_idxs: np.ndarray = np.arange(len(X_src_paths))
    y_src_labels: np.ndarray = le.fit_transform([label for label in y_src_labels])
    # Generate train-test split
    X_src_train_idxs, X_src_test_idxs, y_src_train, y_src_test = train_test_split(
        X_src_idxs, y_src_labels, random_state=configs.get('random_seed', None)
    )

    # Load target language audio paths and labels
    tgt_language_metadata_df = pd.read_csv(os.path.join(configs['tgt_lang_dir'], 'metadata.csv'))  # .sample(n=10, random_state=2)  # TODO remove sampling
    X_tgt_paths: List[str]
    X_tgt_paths, y_tgt_labels = [*zip(*[
        (os.path.join(configs['tgt_lang_dir'], file_name), label)
        for file_name, label in zip(tgt_language_metadata_df.file_name, tgt_language_metadata_df.label)
        if os.path.exists(os.path.join(configs['tgt_lang_dir'], file_name))
    ])]
    y_tgt_labels: np.ndarray = le.transform([label for label in y_tgt_labels])

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

        y_src_train_split = np.array([y_src_train_split]).reshape(-1)
        y_src_test_split = np.array([y_src_test_split]).reshape(-1)
        y_tgt_labels_split = np.array([y_tgt_labels_split]).reshape(-1)

        try:
            assert all(X_src_train[0].shape == x.shape for X in (X_src_train, X_src_test, X_tgt) for x in X)
        except AssertionError:
            # TODO find better solution, this is a temporary fix
            n = min(x.shape[0] for X in (X_src_train, X_src_test, X_tgt) for x in X)
            X_src_train = [x[:n] for x in X_src_train]
            X_src_test = [x[:n] for x in X_src_test]
            X_tgt = [x[:n] for x in X_tgt]

        # Get source language features
        X_src_train_tensor: np.ndarray = np.vstack([x[None, ...] for x in X_src_train])
        X_src_test_tensor: np.ndarray = np.vstack([x[None, ...] for x in X_src_test])
        # Remove nan values
        X_src_train_tensor = np.nan_to_num(X_src_train_tensor)
        X_src_test_tensor = np.nan_to_num(X_src_test_tensor)
        # Do standardisation
        std_scaler_src: StandardScaler = StandardScaler()
        tmp_n_samples: int
        tmp_n_channels: int
        tmp_n_samples, _, tmp_n_channels = X_src_train_tensor.shape
        X_src_train_tensor: np.ndarray = std_scaler_src.fit_transform(
            X_src_train_tensor.reshape(-1, tmp_n_channels)
        ).reshape(tmp_n_samples, -1, tmp_n_channels)
        tmp_n_samples, _, tmp_n_channels = X_src_test_tensor.shape
        X_src_test_tensor: np.ndarray = std_scaler_src.transform(
            X_src_test_tensor.reshape(-1, tmp_n_channels)
        ).reshape(tmp_n_samples, -1, tmp_n_channels)

        # Get target language features
        X_tgt_tensor: np.ndarray = np.vstack([x[None, ...] for x in X_tgt])
        # Remove nan values
        X_tgt_tensor = np.nan_to_num(X_tgt_tensor)
        # Do standardisation
        std_scaler_tgt: StandardScaler = StandardScaler()
        tmp_n_samples, _, tmp_n_channels = X_tgt_tensor.shape
        X_tgt_tensor: np.ndarray = std_scaler_tgt.fit_transform(
            X_tgt_tensor.reshape(-1, tmp_n_channels)
        ).reshape(tmp_n_samples, -1, tmp_n_channels)

        # Apply minority oversampling to balance the source training data
        lbl_ids, counts = np.unique(y_src_train_split, return_counts=True)
        majority_count = counts.max()
        for lbl, count in zip(lbl_ids, counts):
            if count < majority_count:
                n_resample = majority_count - count
                X_src_train_tensor = np.vstack([
                    X_src_train_tensor,
                    X_src_train_tensor[
                        np.random.choice(np.arange(len(X_src_train_tensor))[y_src_train_split == lbl], size=n_resample)
                    ]
                ])
                y_src_train_split = np.concatenate([y_src_train_split, np.full(n_resample, lbl)])

        # For each pooling approach
        for t_pooling in GlobalPooling:
            # Search for best DNN config given the features
            cv = GridSearchCV(
                estimator=KerasClassifier(
                    model=create_nn,
                    loss="binary_crossentropy",
                    optimizer="adam",
                    metrics=['accuracy'],
                    epochs=150,
                    batch_size=8,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[EarlyStopping(monitor='val_accuracy', patience=5)]
                ),
                param_grid={
                    'model__input_shape': [X_src_train_tensor.shape[1:]],
                    'model__pooling': [t_pooling],
                    'model__hidden_units': [
                        [512], [512, 512], [512, 512, 512], [1024], [1024, 1024], [1024, 1024, 1024]
                    ],
                    'model__dropout_rate': [0.1, 0.333],
                    'optimizer__learning_rate': [1e-3, 1e-4]
                }
            )
            cv.fit(X_src_train_tensor, y_src_train_split)

            # Retain best classifier and compute the test scores
            base_cls = cv.best_estimator_

            # Fit adapter version
            adapter = DeepCORAL(
                encoder=create_nn(
                    input_shape=X_src_train_tensor.shape[1:],
                    n_classes=None,
                    hidden_units=cv.best_params_['model__hidden_units'],
                    dropout_rate=cv.best_params_['model__dropout_rate']
                ),
                task=create_nn(
                    pooling=t_pooling,
                    dropout_rate=cv.best_params_['model__dropout_rate']
                ),
                optimizer=Adam(learning_rate=cv.best_params_['optimizer__learning_rate']),
                optimizer_enc=Adam(learning_rate=cv.best_params_['optimizer__learning_rate']),
                loss='binary_crossentropy',
                lambda_=0.1,
                Xt=X_tgt_tensor,
                metrics=['accuracy'],
                validation_split=0.2,
                epochs=150,
                random_state=configs.get('random_seed', None),
                batch_size=8,
                verbose=0,
                callbacks=[EarlyStopping(monitor='val_accuracy', patience=5)]
            )
            adapter.fit(X_src_train_tensor, y_src_train_split)

            for do_adaptation, cls in zip([False, True], [base_cls, adapter]):
                try:
                    # Get predictions
                    if do_adaptation:
                        y_src_test_pred = (cls.predict(X_src_test_tensor) >= 0.5).astype(int).reshape(-1)
                        y_src_test_proba = cls.predict(X_src_test_tensor).reshape(-1)
                        y_tgt_pred = (cls.predict(X_tgt_tensor) >= 0.5).astype(int)
                        y_tgt_proba = cls.predict(X_tgt_tensor).reshape(-1).reshape(-1)
                    else:
                        y_src_test_pred = cls.predict(X_src_test_tensor)
                        y_src_test_proba = cls.predict_proba(X_src_test_tensor)[:, 1]
                        y_tgt_pred = cls.predict(X_tgt_tensor)
                        y_tgt_proba = cls.predict_proba(X_tgt_tensor)[:, 1]

                    # Test classifier (on source data)
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
                except Exception as e:
                    print(e)
                    src_cls_report = ''
                    src_accuracy_score = src_specificity_score = src_auc_score = src_precision_score = src_recall_score = src_fscore = src_support = None
                    src_fpr = src_tpr = src_roc_threshold = src_pr = src_rc = src_pr_rc_threshold = src_confusion_matrix = np.array([None])
                    tgt_cls_report = ''
                    tgt_accuracy_score = tgt_specificity_score = tgt_auc_score = tgt_precision_score = tgt_recall_score = tgt_fscore = tgt_support = None
                    tgt_fpr = tgt_tpr = tgt_roc_threshold = tgt_pr = tgt_rc = tgt_pr_rc_threshold = tgt_confusion_matrix = np.array([None])

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
