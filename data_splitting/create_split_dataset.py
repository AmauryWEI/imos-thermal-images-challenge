#!/usr/bin/python3

# File:         create_split_dataset.py
# Date:         2023/07/08
# Description:  Script to create a training/validation/testing split using the LTD dataset

import argparse
from os import path, makedirs
from shutil import rmtree
from sys import exit
from random import seed

import numpy as np
import pandas as pd

# Define the arguments/options of the script
parser = argparse.ArgumentParser()

# Mandatory arguments
parser.add_argument(
    "metadata_file",
    metavar="metadata_images.csv",
    help="CSV file containing the images metadata",
    type=str,
)

# Optional arguments
parser.add_argument(
    "-o",
    "--output_dir",
    help="Output directory to create the dataset into",
    default="split_dataset",
    type=str,
)

parser.add_argument(
    "-s",
    "--seed",
    help="Seed of the random number generator",
    default=0,
    type=int,
)

parser.add_argument(
    "-q",
    "--quiet",
    help="Run the script with minimal log output",
    action="store_true",
    default=0,
)

TRAINING_RATIO = 0.6
VALIATION_RATIO = 0.2
TESTING_RATIO = 1 - TRAINING_RATIO - VALIATION_RATIO


def clean_create_directory(absolute_directory_path: str, quiet: bool) -> None:
    """Create a directory (and clean it if already existing)

    Parameters
    ----------
    absolute_directory_path: str
        Absolute path to the directory to create
    """
    if path.exists(absolute_directory_path):
        if not quiet:
            print("Cleaning directory : " + absolute_directory_path)
        rmtree(absolute_directory_path)
    else:
        if not quiet:
            print("Creating directory : " + absolute_directory_path)
    makedirs(absolute_directory_path)


def create_split_directories(target_dir: str, quiet: bool) -> tuple[str, str, str]:
    """
    Create the training, validation, and testing directories

    Parameters
    ----------
    target_dir: str
        Absolute path to the target directory

    Returns
    -------
    str
        Training directory absolute path
    str
        Validation directory absolute path
    str
        Testing directory absolute path
    """
    DIRECTORIES_NAMES = ["training", "validation", "testing"]

    directories_list = []
    for directory_name in DIRECTORIES_NAMES:
        directories_list.append(path.join(target_dir, directory_name))
        clean_create_directory(directories_list[-1], quiet=quiet)

    return directories_list


def split_dataset(
    metadata_abs_path: str,
    training_dir: str,
    validation_dir: str,
    testing_dir: str,
    seed: int = 0,
) -> int:
    """
    Split the raw dataset into different training / validation / testing subsets

    Parameters
    ----------
    metadata_abs_path: str
        Absolute path to the CSV file containing the images metadata
    training_dir: str
        Absolute path to the training set directory
    validation_dir: str
        Absolute path to the validation set directory
    testing_dir: str
        Absolute path to the testing set directory
    seed: int
        Random number generator seed

    Returns
    -------
    int
        0 on success ; 1 on failure
    """
    RAW_METADATA_COLUMNS = [
        "Folder name",
        "Clip Name",
        "Image Number",
        "DateTime",
        "Temperature",
        "Humidity",
        "Precipitation",
        "Dew Point",
        "Wind Direction",
        "Wind Speed",
        "Sun Radiation Intensity",
        "Min of sunshine latest 10 min",
    ]

    # Load the contents of the metadata file
    metadata = pd.read_csv(metadata_abs_path, header=0)
    if list(metadata.columns.values) != RAW_METADATA_COLUMNS:
        print("ERROR: Unexpected columns headers ", list(metadata.columns.values))
        return 1

    # Shuffle all of the metadata to randomize the split
    shuffled_metadata = metadata.sample(frac=1, random_state=seed)

    # Define the images split
    images_count = len(shuffled_metadata.index)
    training_images_count = int(TRAINING_RATIO * images_count)
    validation_images_count = int(VALIATION_RATIO * images_count)

    # Split the dataset into training, validation, and testing
    training, validation, testing = np.split(
        shuffled_metadata, [training_images_count, validation_images_count]
    )

    # Save the metadata to CSV
    training.to_csv(path.join(training_dir, "metadata_images.csv"))
    validation.to_csv(path.join(validation, "metadata_images.csv"))
    testing.to_csv(path.join(testing, "metadata_images.csv"))


def main(args: argparse.Namespace) -> int:
    # Set the seeds
    seed(args.seed)
    np.random.seed(args.seed)

    # Convert potentially relative path to absolute path
    metadata_abs_path = path.abspath(args.metadata_file)
    if not args.quiet:
        print("Target file\t: {}".format(metadata_abs_path))

    # Make sure the target CSV file exists
    if not path.exists(metadata_abs_path):
        print("ERROR: File ", metadata_abs_path, " does not exist.")
        return 1
    if not path.isfile(metadata_abs_path):
        print("ERROR: ", metadata_abs_path, " is a directory.")
        return 1

    # Obtain the root directory (which should contain all the subfolders with images)
    dataset_root_dir = path.dirname(metadata_abs_path)

    # Clean/Create the output directory
    output_dir_abs_path = path.abspath(args.output_dir)
    if output_dir_abs_path == dataset_root_dir:
        print("ERROR: Output directory is the raw dataset directory.")
        return 1
    clean_create_directory(output_dir_abs_path, quiet=args.quiet)

    # Create the split dataset folder structure
    training_dir, validation_dir, testing_dir = create_split_directories(
        output_dir_abs_path,
        quiet=args.quiet,
    )

    # Split the dataset into multiple directories
    split_dataset(
        metadata_abs_path,
        training_dir,
        validation_dir,
        testing_dir,
        seed=args.seed,
    )

    return 0


if __name__ == "__main__":
    args = parser.parse_args()
    ret = main(args)
    exit(ret)
