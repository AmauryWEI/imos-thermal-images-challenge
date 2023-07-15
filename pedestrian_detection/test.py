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
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.v2 import functional as F
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("./loaders/")
from pedestrian_dataset import PedestrianDataset

sys.path.append("./models/")
from fasterrcnn_models import (
    FasterRcnnResnet50FpnV2,
    FasterRcnnMobileNetV3LargeFpn,
    FasterRcnnMobileNetV3Large320Fpn,
)
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


def plot_loss(
    ax,
    epoch_losses: np.ndarray,
    batch_losses: np.ndarray,
    loss_name: str,
) -> None:
    textbox_contents = "\n".join(
        (f"Last {loss_name} = {epoch_losses[-1]:.3e}",),
    )
    props = dict(boxstyle="round", facecolor="white", alpha=0.5)
    # Place a text box in bottom left in axes coords
    ax.text(
        0.02,
        0.1,
        textbox_contents,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=props,
    )

    # Plot the epoch losses on the first X axis
    ax.plot(epoch_losses.flatten(), color="C0")
    # Plot the batches loss on a second X axis
    ax2 = ax.twiny()
    ax2.plot(batch_losses.flatten(), alpha=0.25, color="C0")

    # Format the plot
    max_loss = np.amax(epoch_losses)
    ax.set_ylim([0, 1.2 * max_loss])
    ax.set_xlim([0, len(epoch_losses.flatten()) - 1])
    ax2.set_ylim([0, 1.2 * max_loss])
    ax2.set_xlim([0, len(batch_losses.flatten()) - 1])
    ax2.set_xticklabels([])
    ax.grid(visible=True)
    ax.set_xlabel("Epoch [u]")
    ax.set_ylabel(loss_name)
    ax.set_title(loss_name)


def plot_losses(model_tester: ModelTester, model_name: str) -> None:
    losses_fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
    losses_fig.canvas.manager.set_window_title(f"{model_name} - Training Losses")

    all_training_losses = np.empty((0, 5))
    mean_training_losses_epoch = np.empty((0, 5))
    for batch_losses in model_tester.training_losses:
        all_training_losses = np.vstack([all_training_losses, batch_losses])
        mean_training_losses_epoch = np.vstack(
            [mean_training_losses_epoch, np.mean(batch_losses, axis=0)]
        )

    # Plot the different losses on the subplots
    plot_loss(
        ax[0, 0],
        mean_training_losses_epoch[:, 1],
        all_training_losses[:, 1],
        "Classifier Loss",
    )
    plot_loss(
        ax[0, 1],
        mean_training_losses_epoch[:, 2],
        all_training_losses[:, 2],
        "Box Reg Loss",
    )
    plot_loss(
        ax[1, 0],
        mean_training_losses_epoch[:, 3],
        all_training_losses[:, 3],
        "Objectness Loss",
    )
    plot_loss(
        ax[1, 1],
        mean_training_losses_epoch[:, 4],
        all_training_losses[:, 4],
        "Feature Prop Loss",
    )

    losses_fig.tight_layout()

    plt.show(block=True)


def show_random_images_with_predictions(
    dataset: PedestrianDataset,
    predictions: list[dict],
) -> None:
    """
    Show multiple random images of the dataset with the ground truth (green) and the
    predicted bounding boxes (red)

    Parameters
    ----------
    dataset : PedestrianDataset
        Dataset containing the images and bounding boxes
    predictions : list[dict]
        List of predictions (indexes should match the dataset)
    """
    ROWS_COUNT = 4
    COLS_COUNT = 4
    images_fig, axes = plt.subplots(nrows=ROWS_COUNT, ncols=COLS_COUNT, figsize=(14, 8))
    images_fig.canvas.manager.set_window_title("Test Images")

    for row_idx in range(ROWS_COUNT):
        for col_idx in range(COLS_COUNT):
            image_idx = np.random.randint(0, len(dataset))

            tensor_image, target = dataset[image_idx]
            tensor_image = F.convert_dtype(tensor_image, torch.uint8)

            # If objects are in the image, draw their bounding boxes
            if len(target["boxes"]) > 0:
                tensor_image = draw_bounding_boxes(
                    tensor_image,
                    target["boxes"],
                    colors="green",
                    width=2,
                )

            # If objects are in the image, draw their bounding boxes
            if len(predictions[image_idx]["boxes"]) > 0:
                tensor_image = draw_bounding_boxes(
                    tensor_image,
                    predictions[image_idx]["boxes"],
                    colors="red",
                    width=2,
                )

            # Permute the channels for plotting (in: 3 x H x W ; out: H x W x 3)
            axes[row_idx, col_idx].imshow(tensor_image.permute(1, 2, 0))
            # Set image title and turn-off tick labels
            axes[row_idx, col_idx].set_title(
                f"Idx: {image_idx} ; GT: {len(target['boxes'])} ; "
                f"NN: {len(predictions[image_idx]['boxes'])}"
            )
            axes[row_idx, col_idx].set_yticklabels([])
            axes[row_idx, col_idx].set_xticklabels([])

    images_fig.tight_layout()
    plt.show()


def model_from_name(model_name: str) -> Module:
    if model_name == "FasterRcnnResnet50FpnV2":
        return FasterRcnnResnet50FpnV2()
    elif model_name == "FasterRcnnMobileNetV3LargeFpn":
        return FasterRcnnMobileNetV3LargeFpn()
    elif model_name == "FasterRcnnMobileNetV3Large320Fpn":
        return FasterRcnnMobileNetV3Large320Fpn()
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
    plot_losses(model_tester, model_name)
    model_tester.run()
    show_random_images_with_predictions(dataset, model_tester.predictions)


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
