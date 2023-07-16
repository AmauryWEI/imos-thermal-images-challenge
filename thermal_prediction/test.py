#!/usr/bin/python3

# File:         test.py
# Date:         2023/07/11
# Description:  Top-level script to test a Neural Network for the Challenge #1


import argparse
import sys
from os import path

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

sys.path.append("./loaders/")
from thermal_dataset import ThermalDataset

sys.path.append("./models/")
from sample_model import SampleModel
from resnet_models import (
    ResNet50_RgbNoMetadata,
    ResNet50_RgbMetadata,
    ResNet50_RgbMetadataMlp,
    ResNet18_RgbNoMetadata,
    ResNet18_RgbMetadata,
    ResNet18_RgbMetadataMlp,
)
from cnn_models import CnnModel
from mlp_models import MlpModel, MlpModelDateTime
from mobilenet_models import (
    MobileNetV3Small_RgbNoMetadata,
    MobileNetV3Small_RgbMetadata,
    MobileNetV3Small_RgbMetadataMlp,
)
from model_trainer import parameters_count
from model_tester import ModelTester

GRAYSCALE_MODELS = ["SampleModel", "CnnModel", "MlpModel", "MlpModelDateTime"]
RGB_MODELS = [
    "ResNet50",
    "ResNet50Metadata",
    "ResNet50MetadataMlp",
    "ResNet18",
    "ResNet18Metadata",
    "ResNet18MetadataMlp",
    "MobileNetV3Small",
    "MobileNetV3SmallMetadata",
    "MobileNetV3SmallMetadataMlp",
]

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

parser.add_argument("-s", "--save", help="Save predictions", type=bool, default=False)


def plot_losses(model_tester: ModelTester, model_name: str) -> None:
    """
    Plots the training and validation MSE losses of a network ready for testing.

    Parameters
    ----------
    model_tester : ModelTester
        Contains the model, its training and validation losses history
    model_name : str
        Name of the model (for figure and plot titles)
    """
    losses_fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
    losses_fig.canvas.manager.set_window_title(
        f"{model_name} - Training & Validation Losses"
    )

    training_losses = np.array(model_tester.training_losses)
    mean_training_loss_per_epoch = np.mean(training_losses, axis=1)
    validation_losses = np.array(model_tester.validation_losses)
    mean_validation_loss_per_epoch = np.mean(validation_losses, axis=1)
    max_loss = max(
        np.amax(mean_training_loss_per_epoch),
        np.amax(mean_validation_loss_per_epoch),
    )

    textbox_contents = "\n".join(
        (
            f"Last Training MSE Loss    = {mean_training_loss_per_epoch[-1]:.3f}",
            f"Last Validation MSE Loss = {mean_validation_loss_per_epoch[-1]:.3f}",
        )
    )
    props = dict(boxstyle="round", facecolor="white", alpha=0.5)
    # Place a text box in bottom left in axes coords
    ax.text(
        0.8,
        0.9,
        textbox_contents,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    # Plot the mean losses
    ax.plot(mean_training_loss_per_epoch, label="Training", color="C0")
    ax.plot(mean_validation_loss_per_epoch, label="Validation", color="C1")

    # Plot the batch losses
    ax2 = ax.twiny()
    ax2.plot(training_losses.flatten(), label="Training", alpha=0.25, color="C0")
    ax3 = ax.twiny()
    ax3.plot(validation_losses.flatten(), label="Validation", alpha=0.25, color="C1")

    # Format the plot
    ax.legend()
    ax.set_ylim([0, 1.1 * max_loss])
    ax.set_xlim([0, len(mean_training_loss_per_epoch) - 1])
    ax2.set_xlim([0, len(training_losses.flatten()) - 1])
    ax3.set_xlim([0, len(validation_losses.flatten()) - 1])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])
    ax.grid(visible=True)
    ax.set_xlabel("Epoch [u]")
    ax.set_ylabel("MSE Loss [°C^2]")
    ax.set_title(f"{model_name} - Training & Validation Losses")
    losses_fig.tight_layout()

    plt.show(block=True)


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
    elif model_name == "ResNet18Metadata":
        return ResNet18_RgbMetadata()
    elif model_name == "ResNet18MetadataMlp":
        return ResNet18_RgbMetadataMlp()
    elif model_name == "MobileNetV3Small":
        return MobileNetV3Small_RgbNoMetadata()
    elif model_name == "MobileNetV3SmallMetadata":
        return MobileNetV3Small_RgbMetadata()
    elif model_name == "MobileNetV3SmallMetadataMlp":
        return MobileNetV3Small_RgbMetadataMlp()
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def test(
    dataset: Dataset,
    model: Module,
    batch_size: int,
    device: torch.device,
    checkpoint: str,
    model_name: str,
    save_predictions: bool,
) -> None:
    model_tester = ModelTester(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        workers_count=4,
        load_checkpoint_file=checkpoint,
        save_predictions=save_predictions,
        device=device,
        model_name=model_name,
    )
    plot_losses(model_tester, model_name)
    model_tester.run()


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

    # Test the model
    test(
        dataset,
        model,
        batch_size=args.batch,
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
