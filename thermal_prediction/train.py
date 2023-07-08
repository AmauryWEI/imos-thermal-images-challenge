#!/usr/bin/python3

# File:         train.py
# Date:         2023/07/08
# Description:  Top-level script to train a Neural Network for the Challenge #1

import argparse
import sys
from os import path

import torch
from torch.nn import Module
from torch.utils.data import Dataset

sys.path.append("./loaders/")
from thermal_dataset import ThermalDataset

sys.path.append("./models/")
from sample_model import SampleModel
from model_trainer import ModelTrainer

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
    "-q",
    "--quiet",
    help="Run the script with minimal log output",
    action="store_true",
    default=0,
)

parser.add_argument(
    "-e",
    "--epochs",
    help="Number of epochs during training",
    type=int,
    default=10,
)


def train(
    dataset: Dataset,
    model: Module,
    epochs_count: int,
    device: torch.device,
) -> None:
    model_trainer = ModelTrainer(
        model=model,
        dataset=dataset,
        epochs_count=epochs_count,
        batch_size=10,
        learning_rate=1e-3,
        workers_count=1,
        k_folds=5,
        device=device,
    )
    model_trainer.run()


def main(args: argparse.Namespace) -> int:
    device = torch.device("cpu")
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # elif torch.cuda.is_available():
    #     device = torch.device("cuda")

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

    # Load the dataset
    dataset = ThermalDataset(
        metadata_abs_path,
        augmentation=False,
    )

    # Load a model
    model = SampleModel().to(device)
    print(model)

    train(dataset, model, epochs_count=args.epochs, device=device)

    return 0


if __name__ == "__main__":
    args = parser.parse_args()
    ret = main(args)
    exit(ret)
