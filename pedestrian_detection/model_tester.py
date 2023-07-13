# File:         model_tester.py
# Date:         2023/07/13
# Description:  ModelTester class to test neural networks for Challenge #2

from os import path, mkdir

from tqdm import tqdm
import numpy as np
import torch
from torch.nn import Module, L1Loss, CrossEntropyLoss
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
        workers_count: int,
        load_checkpoint_file: str,
        save_predictions: bool = False,
        device: torch.device = torch.device("cpu"),
        model_name: str = "model",
    ) -> None:
        self.__model = model.to(device)
        self.__device = device
        self.__model_name = model_name

        self.__workers_count = workers_count

        self.__cross_entropy_loss_function = CrossEntropyLoss()
        self.__l1_loss_function = L1Loss()
        self.__save_predictions = save_predictions
        self.__predictions = []

        # Loaded from the checkpoint file and accessible for plotting
        self.__training_losses = []
        self.__validation_losses = []
        self.__testing_l1_losses = []
        self.__testing_cross_entropy_losses = []

        # Ensure the directory to save predictions exists
        if self.__save_predictions:
            if not path.exists("predictions"):
                mkdir("predictions")

        self.__data_loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=self.__workers_count,
            collate_fn=collate_fn,
        )

        self.__load_checkpoint(load_checkpoint_file)

    @property
    def training_losses(self) -> list[np.ndarray]:
        return self.__training_losses

    @property
    def validation_losses(self) -> list[np.ndarray]:
        return self.__validation_losses

    @property
    def predictions(self) -> list[dict]:
        return self.__predictions

    def run(self):
        """
        Main entry point to run testing
        """
        # Evaluation mode (the network will output predictions instead of losses)
        self.__model.eval()

        # Turn off gradient to prevent unnecessary computation and save memory
        with torch.no_grad():
            # Prepare a current title on the status bar
            tqdm_iterator = tqdm(self.__data_loader, desc=f"{self.__model_name}")

            for images, targets in tqdm_iterator:
                # Assign tensors to target computing device
                images = list(image.to(self.__device) for image in images)
                # Targets will only be used for loss computation
                targets = [
                    {k: v.to(self.__device) for k, v in t.items()} for t in targets
                ]

                # Inference prediction by model and obtain predictions (instead of loss)
                outputs = self.__model(images)
                outputs = [{k: v for k, v in t.items()} for t in outputs]

                # TODO: Manually compute losses here

                # Store the predictions (only 1 item in the array, as batch_size = 1)
                self.__predictions.append(outputs[0])

            # TODO: Print performance metrics

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


def collate_fn(batch):
    return tuple(zip(*batch))
