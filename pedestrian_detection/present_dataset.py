#!/usr/bin/python3

# File:         present_dataset.py
# Date:         2023/07/12
# Description:  Top-level script to visualize and analyze the Pedestrian Detection dataset

import argparse
from os import path
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.v2 import functional as F

sys.path.append("./loaders/")
from pedestrian_dataset import PedestrianDataset

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
    "-q",
    "--quiet",
    help="Run the script with minimal log output",
    action="store_true",
    default=0,
)

parser.add_argument(
    "-a",
    "--augmented",
    help="Show the augmented version of the dataset (all options enabled)",
    action="store_true",
    default=0,
)


def show_random_images(dataset: PedestrianDataset) -> None:
    """
    Show multiple random images of the dataset with their annotations (bounding boxes)

    Parameters
    ----------
    dataset : PedestrianDataset
        Dataset containing the images and bounding boxes
    """
    ROWS_COUNT = 3
    COLS_COUNT = 3
    images_fig, axes = plt.subplots(nrows=ROWS_COUNT, ncols=COLS_COUNT, figsize=(14, 8))
    images_fig.canvas.manager.set_window_title("Dataset Images")

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
                    colors="yellow",
                    width=2,
                )

            # Permute the channels for plotting (in: 3 x H x W ; out: H x W x 3)
            axes[row_idx, col_idx].imshow(tensor_image.permute(1, 2, 0))
            # Set image title and turn-off tick labels
            axes[row_idx, col_idx].set_title(
                f"Idx: {image_idx} ; {len(target['boxes'])} objects"
            )
            axes[row_idx, col_idx].set_yticklabels([])
            axes[row_idx, col_idx].set_xticklabels([])

    images_fig.tight_layout()


def main(args: argparse.Namespace) -> int:
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

    # Create a dataset from the folders
    dataset = PedestrianDataset(
        data_folders_abs_path,
        augment_crop=args.augmented,
        augment_distort=args.augmented,
        augment_flip=args.augmented,
        quiet=args.quiet,
    )

    show_random_images(dataset)
    plt.show()

    return 0


if __name__ == "__main__":
    args = parser.parse_args()
    ret = main(args)
    exit(ret)
