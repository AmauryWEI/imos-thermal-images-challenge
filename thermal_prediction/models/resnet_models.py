# File:         resnet_model.py
# Date:         2023/07/09
# Description:  Models based on ResNet pre-trained architectures

from torch import Tensor, relu, cat
from torch.nn import Module, Linear
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50_RgbNoMetadata(Module):
    """
    Input:  Image (cols: 384 ; rows: 288) (Tensor 288 x 384) + Metadata (Tensor 9 x 1) (not used)
    Output: Temperature (float)
    """

    def __init__(self):
        super(ResNet50_RgbNoMetadata, self).__init__()

        self.__resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Do not fine-tune the ResNet50 weights
        for param in self.__resnet.parameters():
            param.requires_grad = False

        self.__resnet_preprocess = ResNet50_Weights.DEFAULT.transforms(antialias=False)

        # Output a single temperature value instead of 1000 classes
        self.__resnet.fc = Linear(2048, 1)

    def forward(self, image: Tensor, metadata: Tensor):
        # ResNet forward
        return self.__resnet.forward(self.__resnet_preprocess(image))


class ResNet50_RgbMetadata(Module):
    """
    Input:  Image (cols: 384 ; rows: 288) (Tensor 288 x 384) + Metadata (Tensor 9 x 1)
    Output: Temperature (float)
    """

    def __init__(self):
        super(ResNet50_RgbMetadata, self).__init__()

        # Multi-Layer Perceptron to process the MetaData
        self.__mlp_output_size = 9

        self.__resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Do not fine-tune the ResNet50 weights
        for param in self.__resnet.parameters():
            param.requires_grad = False

        self.__resnet_preprocess = ResNet50_Weights.DEFAULT.transforms(antialias=False)

        self.__mlp = Linear(9, self.__mlp_output_size)

        # The output of ResNet50 is 1000
        self.__fc = Linear(1000 + self.__mlp_output_size, 1)

    def forward(self, image: Tensor, metadata: Tensor):
        # ResNet image processing
        image = self.__resnet_preprocess(image)
        resnet_out = self.__resnet.forward(image)

        # MLP metadata processing
        mlp_out = relu(self.__mlp(metadata))

        # Network Out with ResNet Out and MLP Out (no activation function)
        out = cat((resnet_out, mlp_out), 1)
        # No activation function to output a continuous value
        out = self.__fc(out)

        return out
