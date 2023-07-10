# File:         thermal_dataset.py
# Date:         2023/07/08
# Description:  Definition of the ThermalDataset class to use for Challenge #1

from os import path
from datetime import datetime

import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
)


RAW_METADATA_COLUMNS = [
    "Folder name",
    "Clip Name",
    "Image Number",
    "DateTime",
    "Temperature",
    "Humidity",
    "Precipitation",
    "Dew Point",
    "Wind Direction",
    "Wind Speed",
    "Sun Radiation Intensity",
    "Min of sunshine latest 10 min",
]


class ThermalDataset(Dataset):
    def __init__(
        self,
        metadata_abs_path: str,
        images_abs_path: str,
        grayscale_to_rgb: bool = False,
        normalize: bool = True,
        augment: bool = False,
        quiet: bool = False,
    ) -> None:
        """
        Initialize the ThermalDataset class

        Parameters
        ----------
        metadata_abs_path : str
            Absolute path to the metadata_images.csv file
        images_abs_path : str
            Absolute path to the root folder containing the images subfolders
        normalize: bool, optional
            Normalize the images and metadata, by default True
        augment: bool, optional
            Augment the dataset, by default False
        quiet: bool, optional
            No log output, by default False
        """
        self.__metadata_abs_path = metadata_abs_path
        self.__images_abs_path = images_abs_path
        self.__grayscale_to_rgb = grayscale_to_rgb
        self.__normalize = normalize
        self.__quiet = quiet

        self.__rand_flip = Compose([RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5)])
        self.__rgb_resize = Resize(224, antialias=True)  # 224 x 224 for ResNet50

        # Load the metadata CSV file
        self.__metadata = pd.read_csv(self.__metadata_abs_path)
        if not all(
            item in self.__metadata.columns.values for item in RAW_METADATA_COLUMNS
        ):
            raise RuntimeError(
                f"Unexpected columns headers: {list(self.__metadata.columns.values)}"
            )
        if not self.__quiet:
            print("ThermalDataset: metadata loaded")

        # Convert the string date to day of the year + hour
        self.__create_day_and_hour_columns()

        # Augment the dataset if required
        if augment:
            # TODO: Implement Dataset augmentation
            pass

        # Compute normalization parameters
        if self.__normalize:
            self.__normalize_metadata()

    @property
    def metadata(self) -> pd.DataFrame:
        return self.__metadata

    def __len__(self) -> int:
        return len(self.__metadata.index)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, float]:
        """
        Access an image, its metadata, and its ground truth temperature

        Parameters
        ----------
        index : int
            Target index in the dataset

        Returns
        -------
        tuple[Tensor, Tensor, float]
            Image data, Image metadata, Ground truth Temperature
        """
        image = self.__fetch_image_as_tensor(index).float()
        metadata = self.__metadata_as_tensor(index).float()
        temperature = Tensor([self.__metadata.loc[index]["Temperature"]]).float()

        return image, metadata, temperature

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
            Target grayscale image as a Tensor
        """
        data_frame_row = self.__metadata.loc[index]
        image_abs_path = path.join(
            self.__images_abs_path,
            data_frame_row["Folder name"].astype(str),
            data_frame_row["Clip Name"],
            data_frame_row["Image Number"] + ".jpg",
        )
        if self.__grayscale_to_rgb:
            return self.__rgb_resize(
                self.__rand_flip(read_image(image_abs_path, ImageReadMode.RGB) / 255.0)
            )
        else:
            return self.__rand_flip(
                read_image(image_abs_path, ImageReadMode.GRAY) / 255.0
            )

    def __metadata_as_tensor(self, index: int) -> Tensor:
        """
        Return the metadata at a specific index in the dataset

        Parameters
        ----------
        index : int
            Target index in the dataset

        Returns
        -------
        Tensor
            Metadata tensor with columns
            "Humidity", "Precipitation", "Dew Point", "Wind Direction", "Wind Speed",
            "Sun Radiation Intensity", "Min of sunshine latest 10 min", "Day", "Hour"
        """
        return Tensor(self.__metadata.iloc[index, 5:14])

    def __create_day_and_hour_columns(self) -> None:
        if "Day" not in self.__metadata or "Hour" not in self.__metadata:
            # Create empty columns in the DataFrame
            self.__metadata["Day"] = 0
            self.__metadata["Hour"] = 0

            for idx in self.__metadata.index:
                # Convert the "DateTime" string to an actual Python object
                datetime_object = datetime.strptime(
                    self.__metadata.at[idx, "DateTime"], "%Y-%m-%d %H:%M:%S"
                )
                # Find out the first day of the year the image was taken in
                year_start = datetime_object.replace(month=1).replace(day=1)
                # Assign the "Day" column to a number of days (will be in [0 - 365])
                self.__metadata.at[idx, "Day"] = (datetime_object - year_start).days
                # Assign the "Hour" column to a number of minutes (will be in [0 - (24*60 - 1)])
                self.__metadata.at[idx, "Hour"] = (
                    datetime_object.hour * 60 + datetime_object.minute
                )

            # Update the CSV file on the disk (to speed-up for next time)
            self.__metadata.to_csv(self.__metadata_abs_path, index=False)

            if not self.__quiet:
                print("ThermalDataset: 'Day' and 'Hour' columns created")

    def __normalize_metadata(self) -> None:
        # Normalize the "Humidity" column (between 0 and 100 [%])
        self.__metadata["Humidity"] = self.__metadata["Humidity"] / 100

        # Normalize the "Precipitation" column
        mean = self.__metadata["Precipitation"].mean()
        std = self.__metadata["Precipitation"].std()
        self.__metadata["Precipitation"] = (
            self.__metadata["Precipitation"] - mean
        ) / std

        # Normalize the "Dew Point" column
        mean = self.__metadata["Dew Point"].mean()
        std = self.__metadata["Dew Point"].std()
        self.__metadata["Dew Point"] = (self.__metadata["Dew Point"] - mean) / std

        # Normalize the "Wind Direction" column (between 0 and 360 [deg])
        self.__metadata["Wind Direction"] = self.__metadata["Wind Direction"] / 360.0

        # Normalize the "Wind Speed" column
        mean = self.__metadata["Wind Speed"].mean()
        std = self.__metadata["Wind Speed"].std()
        self.__metadata["Wind Speed"] = (self.__metadata["Wind Speed"] - mean) / std

        # Normalize the "Sun Radiation Intensity" column
        mean = self.__metadata["Sun Radiation Intensity"].mean()
        std = self.__metadata["Sun Radiation Intensity"].std()
        self.__metadata["Sun Radiation Intensity"] = (
            self.__metadata["Sun Radiation Intensity"] - mean
        ) / std

        # Normalize the "Min of sunshine latest 10 min" column (between 0 and 10 [min])
        self.__metadata["Min of sunshine latest 10 min"] = (
            self.__metadata["Min of sunshine latest 10 min"] / 10.0
        )

        # Normalize the "Day" column (between 0 and 365 [days])
        self.__metadata["Day"] = self.__metadata["Day"] / 365.0

        # Normalize the "Hour" column (between 0 and 1439 [min])
        self.__metadata["Hour"] = self.__metadata["Hour"] / (24.0 * 60 - 1)

        if not self.__quiet:
            print("ThermalDataset: metadata columns normalized")
