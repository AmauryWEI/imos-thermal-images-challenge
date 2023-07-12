# File:         cnn_models.py
# Date:         2023/07/09
# Description:  Models based on convolutional architectures (CNNs)

from torch import Tensor
from torch.nn import Module
from torchvision.models import ResNet50_Weights, MobileNet_V3_Large_Weights
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
    faster_rcnn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)


class FasterRcnnResnet50FpnV2(Module):
    """
    Faster R-CNN model with a ResNet-50-FPN (Feature Pyramid Network) backbone.

    This network uses pre-trained weights, and outputs 2 classes. This model definition
    was inspired by
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#finetuning-from-a-pretrained-model

    Input:  Image (cols: 384 ; rows: 288) (Tensor 288 x 384) + Target (N x 4)
    Output: Temperature (float)
    """

    def __init__(self):
        super(FasterRcnnResnet50FpnV2, self).__init__()

        self.__model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            num_classes=91,  # Default classes is 91, but replaced in a few lines
            weights_backbone=ResNet50_Weights.DEFAULT,
            trainable_backbone_layers=3,
        )

        # Determine the number of input features for the classifier
        in_features = self.__model.roi_heads.box_predictor.cls_score.in_features

        # Replace the pre-trained head bwith a new one
        self.__model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(
            in_features,
            num_classes=2,
        )

    def forward(self, image: Tensor, targets: dict = None) -> dict:
        return self.__model(image, targets)


class FasterRcnnMobileNetV3LargeFpn(Module):
    """
    Faster R-CNN model with a MobileNetV3Large FPN (Feature Pyramid Network) backbone.

    This network uses pre-trained weights, and outputs 2 classes. This model definition
    was inspired by
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#finetuning-from-a-pretrained-model

    Input:  Image (cols: 384 ; rows: 288) (Tensor 288 x 384) + Target (N x 4)
    Output: Temperature (float)
    """

    def __init__(self):
        super(FasterRcnnMobileNetV3LargeFpn, self).__init__()

        self.__model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
            num_classes=91,  # Default classes is 91, but replaced in a few lines
            weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
            trainable_backbone_layers=3,
        )

        # Determine the number of input features for the classifier
        in_features = self.__model.roi_heads.box_predictor.cls_score.in_features

        # Replace the pre-trained head bwith a new one
        self.__model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(
            in_features,
            num_classes=2,
        )

    def forward(self, image: Tensor, targets: dict = None) -> dict:
        return self.__model(image, targets)
