# File:         model_trainer.py
# Date:         2023/07/08
# Description:  ModelTrainer class to train neural networks for Challenge #1

from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.nn import Module, MSELoss
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, random_split


class ModelTrainer:

    """
    A training worker for neural network models.

    Loads an initial state of a neural network model, train it using some parameters and
    a given dataset. Finally, exports the training result as a checkpoint file.
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
        normalize_images: bool = True,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.__model = model.to(device)
        self.__device = device

        self.__normalize_images = normalize_images
        self.__training_image_mean = 0
        self.__training_image_std = 1

        self.__epochs_count = epochs_count
        self.__batch_size = batch_size
        self.__workers_count = workers_count

        self.__loss_function = MSELoss()
        self.__optimizer = Adam(model.parameters(), lr=learning_rate)

        self.__fold = 0
        self.__epoch = 0
        self.__train_data_loader = None
        self.__validation_data_loader = None

        # Split into training and validation dataset
        self.__k_folds_datasets = []
        if k_folds == 0:
            # Default training to validation ratio: 80% to 20%
            self.__k_folds_datasets = [
                tuple(self.__split_dataset(dataset, (0.8, 0.2), randomize=True))
            ]
        else:
            self.__k_folds_datasets = self.__split_k_folds(
                dataset, k_folds, randomize=True
            )

    def __split_dataset(
        self,
        dataset: Dataset,
        ratios: list[float],
        randomize: bool = False,
    ) -> list[Subset]:
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
            print(f"Fold {self.__fold+1}:")

            # Prepare validation data
            self.__validation_data_loader = DataLoader(
                validation_data,
                batch_size=self.__batch_size,
                num_workers=self.__workers_count,
            )

            # Training
            for self.__epoch in range(0, self.__epochs_count):
                # Prepare training data, reshuffle for each epoch
                self.__train_data_loader = DataLoader(
                    train_data,
                    batch_size=self.__batch_size,
                    num_workers=self.__workers_count,
                    shuffle=True,
                )
                # Compute the mean and stdandard deviation of the training set
                if self.__normalize_images:
                    self.__compute_image_normalization_parameters()

                self.__train()
                self.__validate()

    def __compute_image_normalization_parameters(self):
        images, _, _ = next(iter(self.__train_data_loader))
        # shape of images = [b,c,w,h]
        self.__training_image_mean, self.__training_image_std = images.mean(
            [0, 2, 3]
        ), images.std([0, 2, 3])

    def __train(self):
        """
        Train model weights and track performance
        """
        training_loss = 0.0
        training_losses_count = 0
        transform = transforms.Normalize(
            self.__training_image_mean, self.__training_image_std
        )

        # Turn on training mode to enable gradient computation
        self.__model.train()

        # Iterate training data with status bar
        tqdm_iterator = tqdm(self.__train_data_loader, desc=f"Epoch {self.__epoch}")
        for _, (image, metadata, temperature) in enumerate(tqdm_iterator):
            # Assign tensors to target computing device
            image = image.to(self.__device, dtype=torch.float)
            metadata = metadata.to(self.__device, dtype=torch.float)
            temperature = temperature.to(self.__device)

            # Normalize the image if necessary
            if self.__normalize_images:
                image = transform(image)

            # Set gradient to zero to prevent gradient accumulation
            self.__optimizer.zero_grad()

            # Inference prediction by model
            output = self.__model(image, metadata)

            # Obtain loss function and back propagate to obtain gradient
            loss = self.__loss_function(output, temperature)
            loss.backward()

            # Keep track of the loss
            training_loss = training_loss + loss.item()
            training_losses_count = training_losses_count + 1

            # Weight learning
            self.__optimizer.step()

            # Store performance metrics and update loss on status bar
            tqdm_iterator.set_postfix_str(f"Loss: {loss.item():.3e}", refresh=False)

        mean_loss = training_loss / training_losses_count
        print(f"Epoch {self.__epoch}: Mean training loss {mean_loss:.2f}")

    def __validate(self):
        validation_loss = 0.0
        validation_losses_count = 0
        # Evaluation mode. Disable running mean and variance of batch normalization
        self.model.eval()

        # Turn off gradient to prevent autograd in backward pass, saves memory
        with torch.no_grad():
            for image, metadata, temperature in tqdm(
                self.__validation_data_loader, desc=f"Validation Fold {self.__fold+1}"
            ):
                # Assign tensors to target computing device
                image = image.to(self.__device, dtype=torch.float)
                metadata = metadata.to(self.__device, dtype=torch.float)
                temperature = temperature.to(self.__device)

                # Inference prediction by model and obtain loss
                output = self.model(image)
                loss = self.__loss_function(output, temperature)

                # Keep track of the loss
                validation_loss = validation_loss + loss.item()
                validation_losses_count = validation_losses_count + 1

        # Obtain and save mean performance for this round
        mean_loss = validation_loss / validation_losses_count
        print(f"Epoch {self.__epoch}: Mean validation loss {mean_loss:.2f}")


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
