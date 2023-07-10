# File:         model_tester.py
# Date:         2023/07/10
# Description:  ModelTester class to test neural networks for Challenge #1

from os import path, mkdir

from tqdm import tqdm
from numpy import mean
import torch
from torch.nn import Module, MSELoss
from torch.utils.data import Dataset, DataLoader


class ModelTester:
    """
    A testing worker for neural network models.

    Loads a neural network model, tests it on a given dataset, and computes different
    performance metrics.
    """

    def __init__(
        self,
        model: Module,
        dataset: Dataset,
        batch_size: int,
        workers_count: int,
        load_checkpoint_file: str,
        save_predictions: bool = False,
        device: torch.device = torch.device("cpu"),
        model_name: str = "model",
    ) -> None:
        self.__model = model.to(device)
        self.__device = device
        self.__model_name = model_name

        self.__batch_size = batch_size
        self.__workers_count = workers_count

        self.__loss_function = MSELoss()
        self.__save_predictions = save_predictions

        # Loaded from the checkpoint file and accessible for plotting
        self.__training_losses = []
        self.__validation_losses = []
        self.__testing_losses = []

        # Ensure the directory to save predictions exists
        if self.__save_predictions:
            if not path.exists("predictions"):
                mkdir("predictions")

        self.__data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.__batch_size,
            num_workers=self.__workers_count,
        )

        self.__load_checkpoint(load_checkpoint_file)

    @property
    def training_losses(self) -> list[list[float]]:
        return self.__training_losses

    @property
    def validation_losses(self) -> list[list[float]]:
        return self.__validation_losses

    def run(self):
        """
        Main entry point to run testing
        """

        # Prepare to save predictions
        predictions = []

        # Switch on evaluation mode to disable batch normalisation
        self.__model.eval()

        # Turn off gradient to prevent unnecessary computation and save memory
        with torch.no_grad():
            # Prepare a current title on the status bar
            tqdm_iterator = tqdm(self.__data_loader, desc=f"{self.__model_name}")

            for batch_idx, (image, metadata, temperature) in enumerate(tqdm_iterator):
                # Assign tensors to target computing device
                image = image.to(self.__device, dtype=torch.float)
                metadata = metadata.to(self.__device, dtype=torch.float)
                temperature = temperature.to(self.__device)  # Ground Truth

                output = self.__model(image, metadata)

                predictions.append(output.cpu().detach().numpy())

                loss = self.__loss_function(output, temperature)
                self.__testing_losses.append(loss.item())

            # Print mean performance for this round
            print(f"ModelTester: Mean loss {mean(self.__testing_losses):.2f}")

            # Stack the prediction and save as numpy file
            if self.__save_predictions:
                # TODO: Save predictions
                pass

    def __load_checkpoint(self, checkpoint_file_path: str) -> None:
        print(f"ModelTester: loading checkpoint {checkpoint_file_path}")

        # Make sure the target checkpoint file exists
        if not path.exists(checkpoint_file_path):
            raise RuntimeError(f"Checkpoint {checkpoint_file_path} does not exist.")
        if not path.isfile(checkpoint_file_path):
            raise (f"{checkpoint_file_path} is not a file.")

        checkpoint = torch.load(checkpoint_file_path, map_location=self.__device)
        self.__model.load_state_dict(checkpoint["model_state_dict"])
        self.__training_losses = checkpoint["training_losses"]
        self.__validation_losses = checkpoint["validation_losses"]
