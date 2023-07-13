#!/usr/bin/python3

# File:         test.py
# Date:         2023/07/13
# Description:  Top-level script to test a Neural Network for the Challenge #2

import argparse
from os import path

# Define the arguments/options of the script
parser = argparse.ArgumentParser()

# Mandatory arguments
parser.add_argument(
    "data_folder",
    metavar="testing_data_folder",
    help="Unique folder containing images and labels for testing",
    type=str,
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
    test_data_dir_abs_path = path.abspath(args.data_folder)

    # Make sure the test data folder exists
    if not path.exists(test_data_dir_abs_path):
        print(f"ERROR: {test_data_dir_abs_path} does not exist.")
        return 1
    if not path.isdir(test_data_dir_abs_path):
        print(f"ERROR: {test_data_dir_abs_path} is not a folder.")
        return 1

    return 0


if __name__ == "__main__":
    args = parser.parse_args()
    ret = main(args)
    exit(ret)
