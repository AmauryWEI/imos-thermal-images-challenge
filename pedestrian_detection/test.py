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

sys.path.append("./models/")
from fasterrcnn_models import FasterRcnnResnet50FpnV2, FasterRcnnMobileNetV3LargeFpn
from model_trainer import parameters_count


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

parser.add_argument(
    "-m",
    "--model",
    help="Model to use",
    type=str,
    default="FasterRcnnResnet50FpnV2",
)


def model_from_name(model_name: str) -> Module:
    if model_name == "FasterRcnnResnet50FpnV2":
        return FasterRcnnResnet50FpnV2()
    elif model_name == "FasterRcnnMobileNetV3LargeFpn":
        return FasterRcnnMobileNetV3LargeFpn()
    else:
        raise ValueError(f"Unknown model name: {model_name}")


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

    # Load a model
    model = model_from_name(args.model).to(device)
    if not args.quiet:
        total_params, trainable_params = parameters_count(model)
        print(f"\nModel: {args.model}")
        print(
            f"Parameters: {total_params} ; Trainable: {trainable_params} "
            f"({trainable_params / total_params * 100:.4f} [%])\n"
        )
        print(model)

    return 0


if __name__ == "__main__":
    args = parser.parse_args()
    ret = main(args)
    exit(ret)
