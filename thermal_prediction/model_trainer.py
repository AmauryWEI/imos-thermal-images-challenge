# File:         model_trainer.py
# Date:         2023/07/08
# Description:  ModelTrainer class to train neural networks for Challenge #1

from os import path, makedirs

import numpy as np
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.nn import Module, MSELoss, L1Loss
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, random_split


class ModelTrainer:
    """
    A training worker for neural network models used in challenge #1

    Loads an initial state of a neural network model, trains it using some parameters
    and a given dataset. After each epoch, the model weights and losses (training &
    validation) are exported in a checkpoint file.
    """

    def __init__(
        self,
        model: Module,
        dataset: Dataset,
        epochs_count: int,
        batch_size: int,
        learning_rate: int,
        workers_count: int,
        k_folds: int,
        device: torch.device = torch.device("cpu"),
        load_checkpoint_file: str = "",
        model_name: str = "model",
        checkpoints_dir: str = "checkpoints",
    ) -> None:
        self.__model = model.to(device)
        self.__device = device

        self.__starting_epoch = 0  # Used to potentially resume training from checkpoint
        self.__epochs_count = epochs_count
        self.__batch_size = batch_size
        self.__workers_count = workers_count

        self.__loss_function = MSELoss()
        self.__mae_loss_function = L1Loss()
        self.__optimizer = Adam(model.parameters(), lr=learning_rate)

        self.__fold = 0
        self.__epoch = 0
        self.__train_data_loader = None
        self.__validation_data_loader = None

        # Stored globally to keep track of all losses in each checkpoint
        self.__training_losses = []
        self.__validation_losses = []

        self.__model_name = model_name  # Checkpoints will be saved with this name
        self.__checkpoints_dir = path.abspath(checkpoints_dir)
        if not path.exists(self.__checkpoints_dir):
            makedirs(self.__checkpoints_dir)

        # Split into training and validation dataset
        self.__k_folds_datasets = []
        if k_folds < 2:
            print("ModelTrainer: k_folds < 2 => single 80/20 training/validation ratio")
            # Default training to validation ratio: 80% to 20%
            self.__k_folds_datasets = [
                tuple(self.__split_dataset(dataset, (0.8, 0.2), randomize=True))
            ]
        else:
            self.__k_folds_datasets = self.__split_k_folds(
                dataset, k_folds, randomize=True
            )

        # Just before starting, load a checkpoint file if available
        if len(load_checkpoint_file) > 0:
            self.__load_checkpoint(load_checkpoint_file)

    def __split_dataset(
        self,
        dataset: Dataset,
        ratios: list[float],
        randomize: bool = False,
    ) -> list[Subset]:
        """
        Split a unique dataset into multiple subset datasets

        Parameters
        ----------
        dataset : Dataset
            Main dataset to split
        ratios : list[float]
            Ratios (normalized or not) defining the dataset split (variable length)
        randomize : bool, optional
            Split the main dataset randomly, by default False

        Returns
        -------
        list[Subset]
            Split datasets

        Raises
        ------
        ValueError
            Invalid splitting ratios
        """
        if len(ratios) == 0:
            raise ValueError("Ratios cannot be an empty list")
        if any(r < 0 for r in ratios):
            raise ValueError(f"Each ratio must be in ]0 - 1] ; got {ratios}")

        ratios_sum = sum(ratios)
        normalized_ratios = [r / ratios_sum for r in ratios]

        split_datasets = []

        if randomize:
            split_datasets = random_split(dataset, normalized_ratios)
        else:
            split_datasets = []
            start_idx = 0
            for r in normalized_ratios[:-1]:
                num_data = int(len(dataset) * r)  # type: ignore
                split_datasets.append(
                    Subset(dataset, range(start_idx, start_idx + num_data))
                )
                start_idx += num_data

            # Last subset takes the rest, not using last ratio due to possible rounding difference
            last_subset = Subset(dataset, range(start_idx, len(dataset)))
            split_datasets.append(last_subset)

        return split_datasets

    def __split_k_folds(
        self,
        dataset: Dataset,
        k_folds: int,
        randomize: bool = False,
    ) -> list[tuple[Dataset, Dataset]]:
        """
        Split a unique dataset into multiple k-folds cross-validation datasets

        Parameters
        ----------
        dataset : Dataset
            Main dataset to split
        k_folds : int
            Number of folds for cross-validation
        randomize : bool, optional
            Split the main dataset randomly, by default False

        Returns
        -------
        list[tuple[Dataset, Dataset]]
            Lists of k (training, validation) datasets

        Raises
        ------
        ValueError
            Invalid number of folds
        """
        if k_folds < 2:
            raise ValueError(f"Number of folds must >= 2 ; got {k_folds}")

        k_fold_datasets: list[tuple[Dataset, Dataset]] = []

        # Create k non-overlapping subsets
        sub_datasets = self.__split_dataset(dataset, [1] * k_folds, randomize=randomize)

        for fold in range(k_folds):
            training_set = ConcatDataset(
                [sub_datasets[i] for i in range(k_folds) if i != fold]
            )
            validation_set = sub_datasets[fold]
            k_fold_datasets.append((training_set, validation_set))

        return k_fold_datasets

    def run(self):
        """
        Main entry point to start training
        """
        for self.__fold, (train_data, validation_data) in enumerate(
            self.__k_folds_datasets
        ):
            print(f"ModelTrainer: Fold {self.__fold}:")

            # Prepare validation data
            self.__validation_data_loader = DataLoader(
                validation_data,
                batch_size=self.__batch_size,
                num_workers=self.__workers_count,
            )

            # Training
            for self.__epoch in range(
                self.__starting_epoch,
                self.__starting_epoch + self.__epochs_count,
            ):
                # Prepare training data, reshuffle for each epoch
                self.__train_data_loader = DataLoader(
                    train_data,
                    batch_size=self.__batch_size,
                    num_workers=self.__workers_count,
                    shuffle=True,
                )

                training_losses = self.__train()
                validation_losses = self.__validate()

                # Update the history of training & validation losses for this fold
                self.__training_losses.append(training_losses)
                self.__validation_losses.append(validation_losses)

                # Save the checkpoint (including training & validation losses)
                self.__save_checkpoint()

    def __train(self) -> list[float]:
        """
        Train model weights and track performance

        Returns
        -------
        list[float]
            Loss of each batch during this epoch
        """
        batches_losses = []

        # Turn on training mode to enable gradient computation
        self.__model.train()

        # Iterate training data with status bar
        tqdm_iterator = tqdm(self.__train_data_loader, desc=f"Epoch {self.__epoch}")
        for _, (image, metadata, temperature) in enumerate(tqdm_iterator):
            # Assign tensors to target computing device
            image = image.to(self.__device, dtype=torch.float)
            metadata = metadata.to(self.__device, dtype=torch.float)
            temperature = temperature.to(self.__device)

            # Set gradient to zero to prevent gradient accumulation
            self.__optimizer.zero_grad()

            # Inference prediction by model
            output = self.__model(image, metadata)

            # Obtain loss function and back propagate to obtain gradient
            loss = self.__loss_function(output, temperature)
            loss.backward()

            # Keep track of the loss
            batches_losses.append(loss.item())

            # Weight learning
            self.__optimizer.step()

            # Store performance metrics and update loss on status bar
            tqdm_iterator.set_postfix_str(f"MSE Loss: {loss.item():.3e}")

        print(
            f"Epoch {self.__epoch}: Mean training MSE Loss {np.mean(batches_losses):.4f}"
        )

        return batches_losses

    def __validate(self) -> list[float]:
        """
        Validate model weights and track performance (no training)

        Returns
        -------
        list[float]
            Loss of each batch during this validation stage
        """
        batches_losses = []
        batches_mae_losses = []

        # Evaluation mode. Disable running mean and variance of batch normalization
        self.__model.eval()

        # Turn off gradient to prevent autograd in backward pass, saves memory
        tqdm_iterator = tqdm(
            self.__validation_data_loader, desc=f"Validation Fold {self.__fold}"
        )
        with torch.no_grad():
            for image, metadata, temperature in tqdm_iterator:
                # Assign tensors to target computing device
                image = image.to(self.__device, dtype=torch.float)
                metadata = metadata.to(self.__device, dtype=torch.float)
                temperature = temperature.to(self.__device)

                # Inference prediction by model and obtain loss
                output = self.__model(image, metadata)
                loss = self.__loss_function(output, temperature)
                mae_loss = self.__mae_loss_function(output, temperature)

                # Keep track of the loss
                batches_losses.append(loss.item())
                batches_mae_losses.append(mae_loss.item())

                # Store performance metrics and update loss on status bar
                tqdm_iterator.set_postfix_str(f"MSE Loss: {loss.item():.3e}")

        # Print mean performance for this epoch
        print(
            f"Epoch {self.__epoch}: Mean validation MSE Loss: {np.mean(batches_losses):.4f} "
            f"(MAE Loss: {np.mean(batches_mae_losses):.4f})"
        )

        return batches_losses

    def __save_checkpoint(self) -> None:
        """
        Save the model weights, optimizer states, training losses and validation losses
        inside a checkpoint file.
        """
        target_checkpoint_path = path.join(
            self.__checkpoints_dir,
            f"{self.__model_name}_fold-{self.__fold}_epoch-{self.__epoch}.pt",
        )
        torch.save(
            {
                "epoch": self.__epoch,
                "model_state_dict": self.__model.state_dict(),
                "optimizer_state_dict": self.__optimizer.state_dict(),
                "last_training_loss_mean": np.mean(self.__training_losses[-1]),
                "training_losses": self.__training_losses,
                "last_validation_loss_mean": np.mean(self.__validation_losses[-1]),
                "validation_losses": self.__validation_losses,
            },
            target_checkpoint_path,
        )

    def __load_checkpoint(self, checkpoint_file_path: str) -> None:
        """
        Load a checkpoint file (model weights, optimizer weights, losses) to resume
        training a network.

        Parameters
        ----------
        checkpoint_file_path : str
            Absolute path to the checkpoint file to load

        Raises
        ------
        RuntimeError
            Invalid or inexistant checkpoint file
        """
        print(f"ModelTrainer: loading checkpoint {checkpoint_file_path}")

        # Make sure the target checkpoint file exists
        if not path.exists(checkpoint_file_path):
            raise RuntimeError(f"Checkpoint {checkpoint_file_path} does not exist.")
        if not path.isfile(checkpoint_file_path):
            raise RuntimeError(f"{checkpoint_file_path} is not a file.")

        checkpoint = torch.load(checkpoint_file_path, map_location=self.__device)
        self.__starting_epoch = checkpoint["epoch"] + 1
        self.__model.load_state_dict(checkpoint["model_state_dict"])
        self.__optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.__training_losses = checkpoint["training_losses"]
        self.__validation_losses = checkpoint["validation_losses"]


def parameters_count(model: Module) -> tuple[int, int]:
    """
    Compute the numbers of parameters in a model

    Parameters
    ----------
    model : Module
        Neural network model

    Returns
    -------
    tuple[int, int]
        Total count of parameters, Trainable parameters count
    """
    return [
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ]
