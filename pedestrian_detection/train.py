#!/usr/bin/python3

# File:         train.py
# Date:         2023/07/13
# Description:  Top-level script to train a Neural Network for the Challenge #2

import argparse
import sys
from os import path
from typing import Optional

import torch
from torch.nn import Module
from torch.utils.data import Dataset, ConcatDataset

sys.path.append("./loaders/")
from pedestrian_dataset import PedestrianDataset

sys.path.append("./models/")
from fasterrcnn_models import (
    FasterRcnnResnet50FpnV2,
    FasterRcnnMobileNetV3LargeFpn,
    FasterRcnnMobileNetV3Large320Fpn,
)
from retinanet_models import RetinaNetResnet50FpnV2
from model_trainer import ModelTrainer, parameters_count

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
    "-v",
    "--validation",
    help="Unique folder containing images and labels for validation",
    type=str,
    default="",
)

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

parser.add_argument(
    "-a",
    "--augment",
    help="Augment the dataset",
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

parser.add_argument(
    "-b",
    "--batch",
    help="Batch size",
    type=int,
    default=2,
)

parser.add_argument(
    "-c",
    "--checkpoint",
    help="Initial model checkpoint to load before training",
    type=str,
    default="",
)

parser.add_argument(
    "-lr",
    "--learning_rate",
    help="Learning rate (for AdamW optimizer)",
    type=float,
    default=1e-4,
)

parser.add_argument(
    "-tl",
    "--trainable_layers",
    help="Number of trainable layers in the CNN backbone",
    type=int,
    default=1,
)


def model_from_name(model_name: str, trainable_layers: int) -> Module:
    if model_name == "FasterRcnnResnet50FpnV2":
        return FasterRcnnResnet50FpnV2(trainable_layers)
    elif model_name == "FasterRcnnMobileNetV3LargeFpn":
        return FasterRcnnMobileNetV3LargeFpn(trainable_layers)
    elif model_name == "FasterRcnnMobileNetV3Large320Fpn":
        return FasterRcnnMobileNetV3Large320Fpn(trainable_layers)
    elif model_name == "RetinaNetResnet50FpnV2":
        return RetinaNetResnet50FpnV2(trainable_layers)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def train(
    dataset: Dataset,
    dataset_validation: Optional[Dataset],
    model: Module,
    epochs_count: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    checkpoint: str,
    model_name: str,
) -> None:
    model_trainer = ModelTrainer(
        model=model,
        dataset=dataset,
        dataset_validation=dataset_validation,
        epochs_count=epochs_count,
        batch_size=batch_size,
        learning_rate=learning_rate,
        workers_count=4,
        device=device,
        load_checkpoint_file=checkpoint,
        model_name=model_name,
    )
    model_trainer.run()


def main(args: argparse.Namespace) -> int:
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        # device = torch.device("mps") # Disabled because of an issue on macOS
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")

    if not args.quiet:
        print(f"Data folders ({len(args.data_folders)}): {args.data_folders}")
        print(f"Validation folder: {args.validation}")

    # Make sure each folder exists
    data_folders_abs_path = [path.abspath(f) for f in args.data_folders]
    for folder in data_folders_abs_path:
        if not path.exists(folder):
            print(f"ERROR: {folder} does not exist.")
            return 1
        if not path.isdir(folder):
            print(f"ERROR: {folder} is not a folder.")
            return 1

    validation_folder_abs_path = ""
    if args.validation != "":
        validation_folder_abs_path = path.abspath(args.validation)
        if not path.exists(validation_folder_abs_path):
            print(f"ERROR: {validation_folder_abs_path} does not exist.")
            return 1
        if not path.isdir(validation_folder_abs_path):
            print(f"ERROR: {validation_folder_abs_path} is not a folder.")
            return 1

    # Load the dataset
    dataset = PedestrianDataset(data_folders_abs_path, quiet=args.quiet)
    dataset_validation = (
        PedestrianDataset([validation_folder_abs_path], quiet=args.quiet)
        if validation_folder_abs_path != ""
        else None
    )

    # Perform data augmentation
    if args.augment:
        augmented_dataset = PedestrianDataset(
            data_folders_abs_path,
            augment=True,
            quiet=args.quiet,
        )
        dataset = ConcatDataset([dataset, augmented_dataset])

    # Load a model
    model = model_from_name(args.model, args.trainable_layers).to(device)
    if not args.quiet:
        total_params, trainable_params = parameters_count(model)
        print(f"\nModel: {args.model}")
        print(
            f"Parameters: {total_params} ; Trainable: {trainable_params} "
            f"({trainable_params / total_params * 100:.4f} [%])\n"
        )
        print(model)

    # Train the model
    train(
        dataset,
        dataset_validation,
        model,
        epochs_count=args.epochs,
        batch_size=args.batch,
        learning_rate=args.learning_rate,
        device=device,
        checkpoint=args.checkpoint,
        model_name=f"args.model_layers-{args.trainable_layers}",
    )

    return 0


if __name__ == "__main__":
    args = parser.parse_args()
    ret = main(args)
    exit(ret)
