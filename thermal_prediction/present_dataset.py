#!/usr/bin/python3

# File:         present_dataset.py
# Date:         2023/07/09
# Description:  Top-level script to visualize and analyze a dataset

import argparse
import sys
from os import path

import numpy as np
import matplotlib.pyplot as plt

sys.path.append("./loaders/")
from thermal_dataset import ThermalDataset

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


def plot_histogram(
    ax,
    bins_count: int,
    dataset: ThermalDataset,
    column_name: str,
    unit: str,
) -> None:
    # Present Temperature data (ground truth)
    min_value = dataset.metadata[column_name].min()
    max_value = dataset.metadata[column_name].max()
    mean_value = dataset.metadata[column_name].mean()
    print(
        f"{column_name}\tMin: {min_value:.4f}\tMax: {max_value:.4f}\tMean: {mean_value:.4f}"
    )
    hist, bins = np.histogram(
        dataset.metadata[column_name],
        bins=bins_count,
        # density=True,
    )
    ax.stairs(hist, bins, fill=True)
    ax.set_xlabel(f"{column_name} [{unit}]")
    ax.set_ylabel("Probability")
    ax.set_title(
        f"Min: {min_value:.2f} / Max: {max_value:.2f} / Mean: {mean_value:.2f}"
    )


def show_metadata_distribution(dataset: ThermalDataset) -> None:
    metadata_fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14, 8))
    metadata_fig.canvas.manager.set_window_title("Metadata")

    temperature_fig, ax = plt.subplots(figsize=(10, 8))
    temperature_fig.canvas.manager.set_window_title("Temperature")

    plot_histogram(ax, 50, dataset, "Temperature", "°C")
    plot_histogram(axes[0, 0], 50, dataset, "Humidity", "%")
    plot_histogram(axes[0, 1], 50, dataset, "Precipitation", "mm")
    plot_histogram(axes[0, 2], 50, dataset, "Dew Point", "°C")
    plot_histogram(axes[1, 0], 50, dataset, "Wind Direction", "deg")
    plot_histogram(axes[1, 1], 50, dataset, "Wind Speed", "m/s")
    plot_histogram(axes[1, 2], 50, dataset, "Sun Radiation Intensity", "W/m2")
    plot_histogram(axes[2, 0], 50, dataset, "Min of sunshine latest 10 min", "min")
    plot_histogram(axes[2, 1], 50, dataset, "Day", "u")
    plot_histogram(axes[2, 2], 50, dataset, "Hour", "u")

    metadata_fig.tight_layout()
    temperature_fig.tight_layout()


def main(args: argparse.Namespace) -> int:
    # Convert potentially relative path to absolute path
    metadata_abs_path = path.abspath(args.metadata_file)
    if not args.quiet:
        print("Metadata file\t: {}".format(metadata_abs_path))
    images_dir_abs_path = path.abspath(args.images_dir)
    if not args.quiet:
        print("Images folder\t: {}".format(images_dir_abs_path))

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

    # Load the dataset
    dataset = ThermalDataset(
        metadata_abs_path,
        images_abs_path=images_dir_abs_path,
        grayscale_to_rgb=True,
        normalize=False,
        augment=False,
    )

    show_metadata_distribution(dataset)

    plt.show()

    return 0


if __name__ == "__main__":
    args = parser.parse_args()
    ret = main(args)
    exit(ret)
