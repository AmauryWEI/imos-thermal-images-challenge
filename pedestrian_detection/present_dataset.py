#!/usr/bin/python3

# File:         present_dataset.py
# Date:         2023/07/12
# Description:  Top-level script to visualize and analyze the Pedestrian Detection dataset

import argparse
from os import path

# Define the arguments/options of the script
parser = argparse.ArgumentParser()

# Mandatory arguments
parser.add_argument(
    "data_folders",
    metavar="data_folders",
    nargs="+",
    help="Folders containing images and labels",
    default=[],
)

# Optional arguments
parser.add_argument(
    "-q",
    "--quiet",
    help="Run the script with minimal log output",
    action="store_true",
    default=0,
)


def main(args: argparse.Namespace) -> int:
    if not args.quiet:
        print(f"Data folders ({len(args.data_folders)}): {args.data_folders}")

    # Make sure each folder exists
    data_folders_abs_path = [path.abspath(f) for f in args.data_folders]
    for folder in data_folders_abs_path:
        if not path.exists(folder):
            print(f"ERROR: {folder} does not exist.")
            return 1
        if not path.isdir(folder):
            print(f"ERROR: {folder} is not a folder.")
            return 1

    return 0


if __name__ == "__main__":
    args = parser.parse_args()
    ret = main(args)
    exit(ret)
