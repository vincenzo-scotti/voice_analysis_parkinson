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
import os
import sys
import subprocess
import pandas as pd
import math
from argparse import ArgumentParser, Namespace

OUT_DF_COLUMNS = ['file_name', 'label', 'original_file_name', 'start_time', 'end_time']


def main(args: Namespace):
    # Create dest dir if not exists
    if not os.path.exists(args.dest_dir_path):
        os.mkdir(args.dest_dir_path)
    # Read metadata
    metadata_df = pd.read_csv(args.csv_metadata_file_path)
    # Output metadata
    metadata = []
    # Iterate file groups
    for _, grp in metadata_df.groupby('file_name'):
        for idx, (_, row) in enumerate(grp.iterrows()):
            # Prepare output file path
            output_file_name = (
                    os.path.splitext(row.file_name)[0] + f'_{str(idx).zfill(3)}' + '.wav'
            ).replace(os.path.sep, '_')
            output_file_path = os.path.join(args.dest_dir_path, output_file_name)
            if row[['start_time', 'end_time']].isnull().values.any():
                # If no timing is given simply copy/convert audio file
                subprocess.run([
                    'ffmpeg',
                    '-y',
                    '-i', os.path.join(args.source_dir_path, row.file_name),
                    '-c', 'copy',
                    output_file_path
                ])
            else:
                # Cut audio clip
                subprocess.run([
                    'ffmpeg',
                    '-y',
                    '-i', os.path.join(args.source_dir_path, row.file_name),
                    '-ss', row.start_time,
                    '-to', row.end_time,
                    '-c', 'copy',
                    output_file_path
                ])
            # Save metadata
            metadata.append((output_file_name, row.label, row.file_name, row.start_time, row.end_time))
    # Save metadata
    metadata_df = pd.DataFrame(metadata, columns=OUT_DF_COLUMNS)
    metadata_df.to_csv(
        os.path.join(args.dest_dir_path, 'metadata.csv'),
        mode='a',
        header=not os.path.exists(os.path.join(args.dest_dir_path, 'metadata.csv')),
        index=False
    )

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
