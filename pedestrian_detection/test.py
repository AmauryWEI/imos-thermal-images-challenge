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
from model_tester import ModelTester


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

parser.add_argument(
    "-c",
    "--checkpoint",
    metavar="checkpoint_file",
    help="Model checkpoint to load for testing",
    type=str,
    required=True,
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

parser.add_argument("-s", "--save", help="Save predictions", type=bool, default=False)


def model_from_name(model_name: str) -> Module:
    if model_name == "FasterRcnnResnet50FpnV2":
        return FasterRcnnResnet50FpnV2()
    elif model_name == "FasterRcnnMobileNetV3LargeFpn":
        return FasterRcnnMobileNetV3LargeFpn()
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def test(
    dataset: Dataset,
    model: Module,
    device: torch.device,
    checkpoint: str,
    model_name: str,
    save_predictions: bool,
) -> None:
    model_tester = ModelTester(
        model=model,
        dataset=dataset,
        workers_count=4,
        load_checkpoint_file=checkpoint,
        save_predictions=save_predictions,
        device=device,
        model_name=model_name,
    )
    model_tester.run()


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

    # Test the model
    test(
        dataset,
        model,
        device=device,
        checkpoint=args.checkpoint,
        model_name=args.model,
        save_predictions=args.save,
    )

    return 0


if __name__ == "__main__":
    args = parser.parse_args()
    ret = main(args)
    exit(ret)
