# File:         thermal_dataset.py
# Date:         2023/07/08
# Description:  Definition of the ThermalDataset class to use for Challenge #1

from os import path
from datetime import datetime

import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image


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
        normalize: bool = True,
        augment: bool = False,
    ) -> None:
        """
        Initialize the ThermalDataset class

        Parameters
        ----------
        metadata_abs_path : str
            Absolute path to the metadata_images.csv file
        augmentation : bool, optional
            Augment the dataset, by default False
        """
        self.__dataset_root_dir = path.dirname(metadata_abs_path)
        self.__normalize = normalize

        # Load the metadata CSV file
        self.__metadata = pd.read_csv(metadata_abs_path)
        if list(self.__metadata.columns.values) != RAW_METADATA_COLUMNS:
            raise RuntimeError(
                "Unexpected columns headers: ", list(self.__metadata.columns.values)
            )

        # Convert the string date to day of the year + hour
        self.__create_day_and_hour_columns()

        # Augment the dataset if required
        if augment:
            # TODO: Implement Dataset augmentation
            pass

        # Compute normalization parameters
        if self.__normalize:
            self.__normalize_metadata()
            self.__compute_image_normalization_params()

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
        Fetch a dataset image as a Tensor with index

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
            self.__dataset_root_dir,
            data_frame_row["Folder name"].astype(str),
            data_frame_row["Clip Name"],
            data_frame_row["Image Number"] + ".jpg",
        )
        image = read_image(image_abs_path)

        return image

    def __metadata_as_tensor(self, index: int) -> Tensor:
        return Tensor(self.__metadata.iloc[index, 5:12])

    def __create_day_and_hour_columns(self) -> None:
        # Create empty columns in the DataFrame
        self.__metadata["Day"] = ""
        self.__metadata["Hour"] = ""

        for _, row in self.__metadata.iterrows():
            # Convert the "DateTime" string to an actual Python object
            datetime_object = datetime.strptime(row["DateTime"], "%d/%m/%y %H:%M")
            # Find out the first day of the year the image was taken in
            year_start = datetime_object.replace(month=1).replace(day=1)
            # Assign the "Day" column to a number of days (will be in [0 - 365])
            row["Day"] = (datetime_object - year_start).days
            # Assign the "Hour" column to a number of minutes (will be in [0 - (24*60 - 1)])
            row["Hour"] = datetime_object.hour * 60 + datetime_object.minute

    def __normalize_metadata(self) -> None:
        pass

    def __compute_image_normalization_params(self) -> None:
        pass
