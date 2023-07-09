#!/usr/bin/python3

# File:         present_dataset.py
# Date:         2023/07/09
# Description:  Top-level script to visualize and analyze a dataset

import argparse
import sys
from os import path

sys.path.append("./loaders/")
from thermal_dataset import ThermalDataset

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

# Optional arguments
parser.add_argument(
    "-q",
    "--quiet",
    help="Run the script with minimal log output",
    action="store_true",
    default=0,
)


def main(args: argparse.Namespace) -> int:
    # Convert potentially relative path to absolute path
    metadata_abs_path = path.abspath(args.metadata_file)
    if not args.quiet:
        print("Metadata file\t: {}".format(metadata_abs_path))
    images_dir_abs_path = path.abspath(args.images_dir)
    if not args.quiet:
        print("Images folder\t: {}".format(images_dir_abs_path))

    # Make sure the target CSV file exists
    if not path.exists(metadata_abs_path):
        print("ERROR: File ", metadata_abs_path, " does not exist.")
        return 1
    if not path.isfile(metadata_abs_path):
        print("ERROR: ", metadata_abs_path, " is a directory.")
        return 1

    # Make sure the images root directory exists
    if not path.exists(images_dir_abs_path):
        print("ERROR: Folder ", images_dir_abs_path, " does not exist.")
        return 1
    if not path.isdir(images_dir_abs_path):
        print("ERROR: ", images_dir_abs_path, " is not a directory.")
        return 1

    # Load the dataset
    dataset = ThermalDataset(
        metadata_abs_path,
        images_abs_path=images_dir_abs_path,
        normalize=True,
        augment=False,
    )

    return 0


if __name__ == "__main__":
    args = parser.parse_args()
    ret = main(args)
    exit(ret)
