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
