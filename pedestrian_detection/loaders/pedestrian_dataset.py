# File:         pedestrian_dataset.py
# Date:         2023/07/12
# Description:  Definition of the PedestrianDataset class to use for Challenge #2

from os import listdir, stat
from os.path import isfile, join

from numpy import loadtxt
import torch
from torch import Tensor, tensor
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode


class PedestrianDataset(Dataset):
    def __init__(
        self,
        data_folders: list[str],
        quiet: bool = False,
    ) -> None:
        """
        Initialize the PedestrianDataset class

        For bounding boxes, pedestrians are represented as class 0, and background as
        class 1.

        Parameters
        ----------
        data_folders : list[str]
            List of folders (absolute paths) containing data to display
        quiet: bool, optional
            No log output, by default False
        """
        self.__data_folders = data_folders
        self.__quiet = quiet

        # List of the absolute paths of all images
        self.__images_abs_path = []
        # List of the absolute paths to annotation files
        self.__annotations_abs_path = []
        # List of "targets" per image (dictionary with "boxes" and "labels" arrays)
        self.__targets = []

        # Read the images and bounding boxes from the disk
        self.__load_data_from_folders()

    def __load_data_from_folders(self) -> None:
        """
        Load the images and bounding boxes from self.__data_folders.

        This function will populate self.__images and self.__bounding_boxes
        """
        for folder in self.__data_folders:
            images_found_per_folder = 0

            # List all files (not directories) inside `folder`
            only_files = [f for f in listdir(folder) if isfile(join(folder, f))]
            only_files = sorted(only_files)

            # Store paths of image files (JPG) and annotation files (TXT)
            for file in only_files:
                if file.endswith(".jpg"):
                    images_found_per_folder = images_found_per_folder + 1
                    self.__images_abs_path.append(join(folder, file))
                elif file.endswith(".txt"):
                    self.__annotations_abs_path.append(join(folder, file))

            if not self.__quiet:
                print(
                    f"PedestrianDataset: found {images_found_per_folder} images in {folder}"
                )

        # The number of images and annotation files should be the same
        if len(self.__images_abs_path) != len(self.__annotations_abs_path):
            raise RuntimeError("Number of JPG files != number of TXT files")

        # Make sure that each image (JPG) has its matching annotation file (TXT)
        for idx, image_name in enumerate(self.__images_abs_path):
            if self.__annotations_abs_path[idx][:-3] != image_name[:-3]:
                raise RuntimeError(f"{image_name} does not have annotation file")

        if not self.__quiet:
            print(f"PedestrianDataset: {len(self.__images_abs_path)} images total")

        # Load the bounding boxes stored in each annotation file
        self.__load_targets_from_annotation_files()

    def __load_targets_from_annotation_files(self):
        """
        Load "targets" from self.__annotation_abs_path

        This function will populate self.__targets. Each file contains (column order):
        - Class of the annotated object (in our case, only 0 for pedestrian)
        - X coordinate of the bounding box (col) (normalized based on the image res)
        - Y coordinate of the bounding box (row) (normalized based on the image res)
        - Width of the boudning box (col) (normalized based on the image res)
        - Height of the boudning box (col) (normalized based on the image res)

        Knowing that Pytorch requires non-normalized bounding boxes with [x1, y1, x2, y2],
        the annotations are transformed in this function.
        """

        # Initialize all images with zero bounding boxes
        self.__targets = [{"boxes": [], "labels": []}] * len(
            self.__annotations_abs_path
        )

        for file_idx, file in enumerate(self.__annotations_abs_path):
            # Check if the file is empty first (to avoid Numpy warnings)
            if stat(file).st_size > 0:
                annotations = loadtxt(file)
                if len(annotations.shape) == 2:
                    # Load multiple annotations
                    tensor_annotations = torch.empty(
                        (annotations.shape[0], 4), dtype=torch.float
                    )
                    for annotation_idx, annotation in enumerate(annotations):
                        tensor_annotations[
                            annotation_idx
                        ] = ltd_annotation_to_pytorch_target(annotation)

                    # Store the data inside the class itself
                    self.__targets[file_idx] = {
                        "boxes": tensor_annotations,
                        "labels": tensor([0] * annotations.shape[0], dtype=torch.int64),
                    }
                else:
                    # Load a single annotation
                    tensor_annotation = ltd_annotation_to_pytorch_target(annotations)
                    self.__targets[file_idx] = {
                        "boxes": tensor_annotation.unsqueeze(0),
                        "labels": tensor([0], dtype=torch.int64),
                    }

    def __len__(self) -> int:
        return len(self.__images_abs_path)

    def __getitem__(self, index: int) -> tuple[Tensor, dict]:
        """
        Access an image and its "target" (bounding boxes + classes)

        Parameters
        ----------
        index : int
            Target index in the dataset

        Returns
        -------
        tuple[Tensor, dict]
            Image data (RGB, 3 chanels, pixel values [0-1]), Target dictionary
        """
        return self.__fetch_image_as_tensor(index).float(), self.__targets[index]

    def __fetch_image_as_tensor(self, index: int) -> Tensor:
        """
        Fetch a dataset image as a Tensor with index normalized between [0.0 - 1.0]

        Parameters
        ----------
        index : int
            Target image index in the dataset

        Returns
        -------
        Tensor
            Target image as a Tensor (RGB, 3 channels)
        """
        image_abs_path = self.__images_abs_path[index]
        return read_image(image_abs_path, ImageReadMode.RGB) / 255.0


def ltd_annotation_to_pytorch_target(annotation: list[float]) -> Tensor:
    """
    Convert an annotation from a TXT file (5 columns: class, X_normalized, Y_normalized,
    width_normalized, height_normalized) to a Pytorch "target" bounding box (x1, y1, x2,
    y2).

    Returns
    -------
    Tensor
        Pytorch "target" annotation
    """
    IMAGE_WIDTH = 384
    IMAGE_HEIGHT = 288
    x1 = IMAGE_WIDTH * annotation[1]
    y1 = IMAGE_HEIGHT * annotation[2]
    x2 = x1 + IMAGE_WIDTH * annotation[3]
    y2 = y1 + IMAGE_HEIGHT * annotation[4]
    return tensor([x1, y1, x2, y2], dtype=torch.float)
