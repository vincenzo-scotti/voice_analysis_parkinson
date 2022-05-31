"""
This script takes a csv where each line contains
 - the path of an audio file (the path must be relative to a base directory path, passed to the script);
 - class label.
For each entry of the csv the script uses rnnoise to clean the noise from the audio clip.
All cleaned clips must be stored in a destination directory, passed to the script, together with a new csv.
Each line of the new csv contains
 - a file path of one of the generated clips (the path must be relative to the destination directory),
 - the path of the original clip
 - the two time stamps (a start time and an end time) used to cut the file
 - class label
"""
import os
import sys
import subprocess
import re
from tempfile import NamedTemporaryFile
import pandas as pd
from argparse import ArgumentParser, Namespace

OUT_DF_COLUMNS = ['file_name', 'label']


def main(args: Namespace):
    # Create dest dir if not exists
    if not os.path.exists(args.dest_dir_path):
        os.mkdir(args.dest_dir_path)
    # Read metadata
    metadata_df = pd.read_csv(args.csv_metadata_file_path)
    # Output metadata
    metadata = []
    # Iterate over file paths
    for _, row in metadata_df.iterrows():
        # Prepare output file path
        output_file_name = re.sub(os.path.sep, '_', row.file_path)
        output_file_path = os.path.join(args.dest_dir_path, output_file_name)
        # Temporary files for intermediate output of denoising
        with NamedTemporaryFile(suffix='.wav') as input_wav, NamedTemporaryFile(suffix='.pcm') as output_pcm:
            # Convert source file into compliant input format
            subprocess.run([
                'ffmpeg',
                '-y',
                '-i', os.path.join(args.source_dir_path, row.file_path),
                '-ar', '48000',
                input_wav.name
            ])
            # Apply denoising
            subprocess.run([args.rnnoise, input_wav.name, output_pcm.name])
            # Convert output into desired format
            subprocess.run([
                'ffmpeg',
                '-y',
                '-f', 's16le',
                '-ar', '48000',
                '-ac', '1',
                '-i', output_pcm.name,
                output_file_path
            ])
        # Save metadata
        metadata.append((output_file_name, row.label))
    # Save metadata
    metadata_df = pd.read_csv(metadata, columns=OUT_DF_COLUMNS)
    metadata_df.to_csv(os.path.join(args.dest_dir_path, 'metadata.csv'), index=False)

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
    args_parser.add_argument(
        '--rnnoise', type=str,
        help="Path to the RNNoise demo tool executable."
    )
    # Run experiment
    main(args_parser.parse_args(sys.argv[1:]))
