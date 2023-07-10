#!/usr/bin/python3

# File:         test.py
# Date:         2023/07/11
# Description:  Top-level script to test a Neural Network for the Challenge #1


import argparse
from os import path

import torch

# Define the arguments/options of the script
parser = argparse.ArgumentParser()

# Mandatory arguments
parser.add_argument(
    "metadata_file",
    metavar="metadata_images.csv",
    help="CSV file containing the images metadata",
    type=str,
)

parser.add_argument(
    "images_dir",
    metavar="Image_Dataset path",
    help="Path to the root directory containing all the images folders",
    type=str,
)

parser.add_argument(
    "checkpoint",
    metavar="checkpoint_file",
    help="Model checkpoint to load for testing",
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

parser.add_argument(
    "-m",
    "--model",
    help="Model to use",
    type=str,
    default="sample_model",
)

parser.add_argument(
    "-b",
    "--batch",
    help="Batch size",
    type=int,
    default=32,
)


def main(args: argparse.Namespace) -> int:
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")

    # Convert potentially relative path to absolute path
    metadata_abs_path = path.abspath(args.metadata_file)
    if not args.quiet:
        print(f"Metadata file\t: {metadata_abs_path}")
    images_dir_abs_path = path.abspath(args.images_dir)
    if not args.quiet:
        print(f"Images folder\t: {images_dir_abs_path}")
    model_checkpoint_abs_path = path.abspath(args.checkpoint)
    if not args.quiet:
        print(f"Checkpoint\t: {model_checkpoint_abs_path}\n")

    # Make sure the checkpoint file exists
    if not path.exists(model_checkpoint_abs_path):
        print(f"ERROR: File {model_checkpoint_abs_path} does not exist.")
        return 1
    if not path.isfile(model_checkpoint_abs_path):
        print(f"ERROR: {model_checkpoint_abs_path} is a directory.")
        return 1

    # Make sure the target CSV file exists
    if not path.exists(metadata_abs_path):
        print(f"ERROR: File {metadata_abs_path} does not exist.")
        return 1
    if not path.isfile(metadata_abs_path):
        print(f"ERROR: {metadata_abs_path} is a directory.")
        return 1

    # Make sure the images root directory exists
    if not path.exists(images_dir_abs_path):
        print(f"ERROR: Folder {images_dir_abs_path} does not exist.")
        return 1
    if not path.isdir(images_dir_abs_path):
        print(f"ERROR: {images_dir_abs_path} is not a directory.")
        return 1

    return 0


if __name__ == "__main__":
    args = parser.parse_args()
    ret = main(args)
    exit(ret)
