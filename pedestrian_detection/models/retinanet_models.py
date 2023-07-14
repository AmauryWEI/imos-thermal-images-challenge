# File:         retinanet_models.py
# Date:         2023/07/14
# Description:  Models based on RetinaNet architectures

from torch import Tensor
from torch.nn import Module
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


class RetinaNetResnet50FpnV2(Module):
    """
    RetinaNet model with a ResNet-50-FPN (Feature Pyramid Network) backbone.

    This network uses pre-trained weights, and outputs 2 classes.

    Input:  Image (cols: 384 ; rows: 288) (Tensor 288 x 384) + Target (N x 4)
    Output: Temperature (float)
    """

    def __init__(self):
        super(RetinaNetResnet50FpnV2, self).__init__()

        self.__model = retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT,
            num_classes=91,  # Default classes is 91, but replaced in a few lines
            weights_backbone=ResNet50_Weights.DEFAULT,
            trainable_backbone_layers=3,
        )

        # Replace the classification head to output 2 classes
        default_num_anchors = self.__model.head.classification_head.num_anchors
        self.__model.head.classification_head = RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=default_num_anchors,
            num_classes=2,
        )

    def forward(self, image: Tensor, targets: dict = None) -> dict:
        return self.__model(image, targets)
