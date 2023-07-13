# File:         model_trainer.py
# Date:         2023/07/13
# Description:  ModelTrainer class to train neural networks for Challenge #2

from os import path, makedirs
from typing import Optional
import sys

import numpy as np
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision.ops import nms

sys.path.append("./utils")
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
from engine import _get_iou_types


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
        dataset_validation: Optional[Dataset],
        epochs_count: int,
        batch_size: int,
        learning_rate: int,
        workers_count: int,
        device: torch.device = torch.device("cpu"),
        load_checkpoint_file: str = "",
        model_name: str = "model",
        checkpoints_dir: str = "checkpoints",
    ) -> None:
        self.__model = model.to(device)
        self.__device = device

        self.__starting_epoch = 0
        self.__epochs_count = epochs_count
        self.__batch_size = batch_size
        self.__workers_count = workers_count

        self.__optimizer = Adam(model.parameters(), lr=learning_rate)

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

        # Load training and validation datasets
        self.__dataset = dataset
        self.__dataset_validation = dataset_validation

        # Check if we actually have a validation dataset, otherwise create one
        if self.__dataset_validation is None:
            self.__dataset, self.__dataset_validation = self.__split_dataset(
                dataset, (0.8, 0.2), randomize=True
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

    def run(self):
        """
        Main entry point to start training
        """
        # Prepare validation data (batch size should be 1 for inference)
        self.__validation_data_loader = DataLoader(
            self.__dataset_validation,
            batch_size=1,
            num_workers=self.__workers_count,
            collate_fn=collate_fn,
            drop_last=True,
        )

        # Training
        for self.__epoch in range(
            self.__starting_epoch,
            self.__starting_epoch + self.__epochs_count,
        ):
            # Prepare training data, reshuffle for each epoch
            self.__train_data_loader = DataLoader(
                self.__dataset,
                batch_size=self.__batch_size,
                num_workers=self.__workers_count,
                shuffle=True,
                collate_fn=collate_fn,
            )

            training_losses = self.__train()
            validation_losses = self.__validate()

            # Update the history of training & validation losses
            self.__training_losses.append(training_losses)
            self.__validation_losses.append(validation_losses)

            # Save the checkpoint (including training & validation losses)
            self.__save_checkpoint()

    def __train(self) -> np.ndarray:
        """
        Train model weights and track performance

        Returns
        -------
        ndarray (batches x 5)
            Losses of each batch during this epoch
            (combined, classification, localization, objectness, region proposal)
        """
        losses = np.zeros((0, 5))

        # Training -> gradient computation, model outputs losses directly
        self.__model.train()

        # Iterate training data with status bar
        tqdm_iterator = tqdm(self.__train_data_loader, desc=f"Epoch {self.__epoch}")
        for _, (images, targets) in enumerate(tqdm_iterator):
            # Assign tensors to target computing device
            images = list(image.to(self.__device) for image in images)
            targets = [{k: v.to(self.__device) for k, v in t.items()} for t in targets]

            # Set gradient to zero to prevent gradient accumulation
            self.__optimizer.zero_grad()

            # Inference prediction by model
            loss_dict = self.__model(images, targets)

            # Obtain loss function and back propagate to obtain gradient
            combined_loss = sum(loss for loss in loss_dict.values())
            combined_loss.backward()

            # Manually track the individual losses (to print epoch summary)
            loss_classifier = loss_dict["loss_classifier"].item()
            loss_box_reg = loss_dict["loss_box_reg"].item()
            loss_objectness = loss_dict["loss_objectness"].item()
            loss_rpn_box_reg = loss_dict["loss_rpn_box_reg"].item()

            # Update the global losses array
            losses = np.vstack(
                [
                    losses,
                    [
                        combined_loss.item(),
                        loss_classifier,
                        loss_box_reg,
                        loss_objectness,
                        loss_rpn_box_reg,
                    ],
                ]
            )

            # Weight learning
            self.__optimizer.step()

            # Store performance metrics and update loss on status bar
            tqdm_iterator.set_postfix_str(f"Combined Loss: {combined_loss.item():.3e}")

        print(
            f"Epoch {self.__epoch}: Combined loss {np.mean(losses[:, 0]):.4f} ; "
            f"Classification loss {np.mean(losses[:, 1]):.4f} ; "
            f"Localization loss {np.mean(losses[:, 2]):.4f} ; "
            f"Objectness loss {np.mean(losses[:, 3]):.4f} ; "
            f"Region Proposal loss {np.mean(losses[:, 4]):.4f}"
        )

        return losses

    def __validate(self) -> list[float]:
        """
        Validate model weights and track performance (no training)

        Returns
        -------
        list[float]
            Loss of each batch during this validation stage
        """
        # Convert to COCO format for Pytorch vision to compute mean Average Precision (mAP)
        coco = get_coco_api_from_dataset(self.__validation_data_loader.dataset)
        iou_types = _get_iou_types(self.__model)
        coco_evaluator = CocoEvaluator(coco, iou_types)

        batches_losses = []

        # Evaluation mode (the network will output predictions instead of losses)
        self.__model.eval()

        # Turn off gradient to prevent autograd in backward pass, saves memory
        tqdm_iterator = tqdm(
            self.__validation_data_loader, desc=f"Validation {self.__epoch}"
        )
        with torch.no_grad():
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

                # Perform Non-Maximum Suppression (NMS) on the bounding boxes
                bboxes_idx_to_keep = nms(
                    boxes=outputs[0]["boxes"],
                    scores=outputs[0]["scores"],
                    iou_threshold=0.3,
                )
                final_outputs = [
                    {k: v[bboxes_idx_to_keep] for k, v in outputs[0].items()}
                ]

                # Compute mean Average Precision (mAP) using COCO evaluator
                res = {
                    target["image_id"].item(): output
                    for target, output in zip(targets, final_outputs)
                }
                coco_evaluator.update(res)

        # Print mean performance for this epoch
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

        return batches_losses

    def __save_checkpoint(self) -> None:
        target_checkpoint_path = path.join(
            self.__checkpoints_dir,
            f"{self.__model_name}_epoch-{self.__epoch}.pt",
        )
        torch.save(
            {
                "epoch": self.__epoch,
                "model_state_dict": self.__model.state_dict(),
                "optimizer_state_dict": self.__optimizer.state_dict(),
                "training_losses": self.__training_losses,
                "validation_losses": self.__validation_losses,
            },
            target_checkpoint_path,
        )

    def __load_checkpoint(self, checkpoint_file_path: str) -> None:
        print(f"ModelTrainer: loading checkpoint {checkpoint_file_path}")

        # Make sure the target checkpoint file exists
        if not path.exists(checkpoint_file_path):
            raise RuntimeError(f"Checkpoint {checkpoint_file_path} does not exist.")
        if not path.isfile(checkpoint_file_path):
            raise (f"{checkpoint_file_path} is not a file.")

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


def collate_fn(batch):
    return tuple(zip(*batch))
