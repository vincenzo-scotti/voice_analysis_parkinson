"""
This script takes a csv where each line contains
 - the path of an audio file (the path must be relative to a base directory path, passed to the script);
 - two time stamps (a start time and an end time);
 - class label.
(File paths can appear multiple times.)
For each entry of the csv the script uses ffmpeg to extract an audio clip between the start and end time stamps.
All generated clips must be stored in a destination directory, passed to the script, together with a new csv.
Each line of the new csv contains
 - a file path (absolute path) of one of the generated clips,
 - the path of the original clip
 - the two time stamps (a start time and an end time) used to cut the file
 - class label
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
        '--csv_metadata_file_path', type=str,
        help="Path to CSV file with the input audio clips."
    )
    args_parser.add_argument(
        '--source_dir_path', type=str,
        help="Path to the base directory containing all the audio clips."
    )
    args_parser.add_argument(
        '--dest_dir_path', type=str,
        help="Path to the base directory containing all the audio clips."
    )
    # Run experiment
    main(args_parser.parse_args(sys.argv[1:]))
