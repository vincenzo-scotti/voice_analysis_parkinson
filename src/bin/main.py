"""
This is the main script.
The script takes the path to the source directory of the English data set, and that of the Hindi data set.
The directories will contain cleaned and cut audio clips to classify

For each of the considered features the script will train and test a model on the English corpus
and then evaluate that same model on the Hindi corpus.
All evaluation results will be logged to make comparisons between the models results.
"""
import sys
from argparse import ArgumentParser, Namespace


def main(args: Namespace):
    return 0


if __name__ == "__main__":
    # Instantiate argument parser
    args_parser: ArgumentParser = ArgumentParser()
    # Add arguments to parser
    args_parser.add_argument(
        '--source_language_data_dir_path', type=str, default='./resources/data/english_corpus/',
        help="Path to the directory with the audio clips in the source language for the experiments."
    )
    args_parser.add_argument(
        '--target_language_data_dir_path', type=str, default='./resources/data/hindi_corpus/',
        help="Path to the directory with the audio clips in the source language for the experiments."
    )
    # Run experiment
    main(args_parser.parse_args(sys.argv[1:]))
