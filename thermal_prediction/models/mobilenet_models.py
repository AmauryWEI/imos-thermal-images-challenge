# File:         mobilenet_models.py
# Date:         2023/07/11
# Description:  Models based on MobileNet v3 pre-trained architectures

from torch import Tensor, cat
from torch.nn import Module, Linear, Identity
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MobileNetV3Small_RgbNoMetadata(Module):
    """
    Model based on a pre-trained MobileNetV3 (Small) architecture, using only an RGB
    input image (the metadata is not used).

    The complete MobileNetV3 model is frozen, except for the last 2 feature blocks. The
    MobileNetV3 output classifier (576 -> 1000) is replaced by a trainable linear layer
    (576 -> 1) to output a unique continuous value (= temperature).

    Inputs:
        - RGB Image (channels: 3 ; cols: 384 ; rows: 288) (Tensor 3 x 224 x 224)
        - Metadata (Tensor 9 x 1) (not used)
    Output:
        - Temperature (float)
    """

    def __init__(self):
        super(MobileNetV3Small_RgbNoMetadata, self).__init__()

        # Load a pre-trained MobileNetV3 Small network
        self.__mobilenet = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT
        )

        # Disable FineTuning on the complete model
        for param in self.__mobilenet.parameters():
            param.requires_grad = False

        # FineTune only the last two MobileNet layers
        for param in self.__mobilenet.features[-2].parameters():
            param.requires_grad = True
        for param in self.__mobilenet.features[-1].parameters():
            param.requires_grad = True

        # Image pre-processing layer (resize, center cropping, ImageNet normalization)
        self.__mobilenet_preprocess = MobileNet_V3_Small_Weights.DEFAULT.transforms(
            antialias=False
        )

        # Modify the classifier block to output a single value (instead of 1000 classes)
        self.__mobilenet.classifier = Linear(576, 1)

    def forward(self, image: Tensor, metadata: Tensor):
        # ResNet forward
        return self.__mobilenet.forward(self.__mobilenet_preprocess(image))


class MobileNetV3Small_RgbMetadata(Module):
    """
    Model based on a pre-trained MobileNetV3 (Small) architecture, using an RGB input image
    and its complete metadata.

    The complete MobileNetV3 model is frozen, except for the last 2 feature blocks. The
    MobileNetV3 output classifier (576 -> 1000) is removed. The 576 MobileNetV3 output
    are concatenated with the metadata (9) and pass through a linear layer (586 -> 1) to
    output a unique continuous value (= temperature).

    Inputs:
        - RGB Image (channels: 3 ; cols: 384 ; rows: 288) (Tensor 3 x 224 x 224)
        - Metadata (Tensor 9 x 1) (not used)
    Output:
        - Temperature (float)
    """

    def __init__(self):
        super(MobileNetV3Small_RgbMetadata, self).__init__()

        #########################################
        # MobileNetV3 Small Network Configuration
        #########################################

        # Load a pre-trained MobileNetV3 Small network
        self.__mobilenet = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT
        )

        # Disable FineTuning on the complete model
        for param in self.__mobilenet.parameters():
            param.requires_grad = False

        # FineTune only the last two MobileNet layers
        for param in self.__mobilenet.features[-2].parameters():
            param.requires_grad = True
        for param in self.__mobilenet.features[-1].parameters():
            param.requires_grad = True

        # Image pre-processing layer (resize, center cropping, ImageNet normalization)
        self.__mobilenet_preprocess = MobileNet_V3_Small_Weights.DEFAULT.transforms(
            antialias=False
        )

        # Replace the default classifier by Identity (to get the raw 576 values)
        self.__mobilenet.classifier = Identity()

        ##############################
        # Network Output with Metadata
        ##############################

        # The output of MobileNet is 576 + 9 metadata parameters
        self.__fc = Linear(576 + 9, 1)

    def forward(self, image: Tensor, metadata: Tensor):
        # ResNet forward
        mobilenet_out = self.__mobilenet.forward(self.__mobilenet_preprocess(image))

        # Network Out with ResNet Out and metadata
        out = cat((mobilenet_out, metadata), 1)
        # No activation function to output a continuous value
        out = self.__fc(out)

        return out
