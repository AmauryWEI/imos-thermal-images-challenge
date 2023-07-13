#!/usr/bin/python3

# File:         test.py
# Date:         2023/07/13
# Description:  Top-level script to test a Neural Network for the Challenge #2

import argparse
from os import path
import sys

import torch
from torch.nn import Module
from torch.utils.data import Dataset

sys.path.append("./loaders/")
from pedestrian_dataset import PedestrianDataset

# Define the arguments/options of the script
parser = argparse.ArgumentParser()

# Mandatory arguments
parser.add_argument(
    "data_folders",
    metavar="data_folders",
    nargs="+",
    help="Folders containing images and labels for testing",
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
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        # device = torch.device("mps") # Disabled because of an issue on macOS
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")

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

    # Load the dataset
    dataset = PedestrianDataset(data_folders_abs_path, quiet=args.quiet)

    return 0


if __name__ == "__main__":
    args = parser.parse_args()
    ret = main(args)
    exit(ret)
