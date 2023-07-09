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
from resnet_models import ResNet50_RgbNoMetadata, ResNet50_RgbMetadata
from cnn_models import CnnModel
from model_trainer import ModelTrainer, parameters_count

GRAYSCALE_MODELS = ["SampleModel", "CnnModel"]
RGB_MODELS = ["ResNet50NoMetadata", "ResNet50"]

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

parser.add_argument(
    "-m",
    "--model",
    help="Model to use",
    type=str,
    default="sample_model",
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
    default=50,
)

parser.add_argument(
    "-c",
    "--checkpoint",
    help="Initial model checkpoint to load before training",
    type=str,
    default="",
)


def requires_rgb(model: str) -> None:
    return True if model in RGB_MODELS else False


def model_from_name(model_name: str) -> Module:
    if model_name == "SampleModel":
        return SampleModel()
    if model_name == "CnnModel":
        return CnnModel()
    elif model_name == "ResNet50NoMetadata":
        return ResNet50_RgbNoMetadata()
    elif model_name == "ResNet50":
        return ResNet50_RgbMetadata()
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def train(
    dataset: Dataset,
    model: Module,
    epochs_count: int,
    batch_size: int,
    normalize_images: bool,
    device: torch.device,
    checkpoint: str,
    model_name: str,
) -> None:
    model_trainer = ModelTrainer(
        model=model,
        dataset=dataset,
        epochs_count=epochs_count,
        batch_size=batch_size,
        learning_rate=1e-3,
        workers_count=1,
        k_folds=5,
        normalize_images=normalize_images,
        device=device,
        load_checkpoint_file=checkpoint,
        model_name=model_name,
    )
    model_trainer.run()


def main(args: argparse.Namespace) -> int:
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")

    # Convert potentially relative path to absolute path
    metadata_abs_path = path.abspath(args.metadata_file)
    if not args.quiet:
        print("Metadata file\t: {}".format(metadata_abs_path))
    images_dir_abs_path = path.abspath(args.images_dir)
    if not args.quiet:
        print("Images folder\t: {}\n".format(images_dir_abs_path))

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

    # Make sure the model is known
    if args.model not in GRAYSCALE_MODELS and args.model not in RGB_MODELS:
        print(f"ERROR: Unknown model {args.model}")
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
        model,
        epochs_count=args.epochs,
        batch_size=args.batch,
        normalize_images=True,
        device=device,
        checkpoint=args.checkpoint,
        model_name=args.model,
    )

    return 0


if __name__ == "__main__":
    args = parser.parse_args()
    ret = main(args)
    exit(ret)
