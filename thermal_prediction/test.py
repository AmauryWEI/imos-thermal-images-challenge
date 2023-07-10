#!/usr/bin/python3

# File:         test.py
# Date:         2023/07/11
# Description:  Top-level script to test a Neural Network for the Challenge #1


import argparse
import sys
from os import path

import torch
from torch.nn import Module

sys.path.append("./loaders/")
from thermal_dataset import ThermalDataset

sys.path.append("./models/")
from sample_model import SampleModel
from resnet_models import (
    ResNet50_RgbNoMetadata,
    ResNet50_RgbMetadata,
    ResNet50_RgbMetadataMlp,
    ResNet18_RgbNoMetadata,
)
from cnn_models import CnnModel
from mlp_models import MlpModel, MlpModelDateTime
from model_trainer import parameters_count

GRAYSCALE_MODELS = ["SampleModel", "CnnModel", "MlpModel", "MlpModelDateTime"]
RGB_MODELS = ["ResNet50", "ResNet50Metadata", "ResNet50MetadataMlp", "ResNet18"]

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


def requires_rgb(model: str) -> None:
    return True if model in RGB_MODELS else False


def model_from_name(model_name: str) -> Module:
    if model_name == "SampleModel":
        return SampleModel()
    if model_name == "CnnModel":
        return CnnModel()
    if model_name == "MlpModel":
        return MlpModel()
    if model_name == "MlpModelDateTime":
        return MlpModelDateTime()
    elif model_name == "ResNet50":
        return ResNet50_RgbNoMetadata()
    elif model_name == "ResNet50Metadata":
        return ResNet50_RgbMetadata()
    elif model_name == "ResNet50MetadataMlp":
        return ResNet50_RgbMetadataMlp()
    elif model_name == "ResNet18":
        return ResNet18_RgbNoMetadata()
    else:
        raise ValueError(f"Unknown model name: {model_name}")


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

    # Load the dataset
    dataset = ThermalDataset(
        metadata_abs_path,
        images_abs_path=images_dir_abs_path,
        grayscale_to_rgb=requires_rgb(args.model),
        normalize=True,
        augment=False,
    )

    # Load a model
    model = model_from_name(args.model).to(device)
    if not args.quiet:
        total_params, _ = parameters_count(model)
        print(f"\nModel: {args.model} (Parameters: {total_params})")
        print(model)

    return 0


if __name__ == "__main__":
    args = parser.parse_args()
    ret = main(args)
    exit(ret)
