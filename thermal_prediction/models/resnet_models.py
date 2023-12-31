# File:         resnet_model.py
# Date:         2023/07/09
# Description:  Models based on ResNet pre-trained architectures

from torch import Tensor, tanh, cat
from torch.nn import Module, Linear, Identity
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights


class ResNet50_RgbNoMetadata(Module):
    """
    Model based on a pre-trained ResNet50 architecture, using only an RGB input image
    (the metadata is not used).

    The complete ResNet50 model is frozen, except for the last convolution block. The
    last ResNet50 linear layer (2048 -> 1000) is replaced by a trainable linear layer
    (2048 -> 1) to output a unique continuous value (= temperature).

    Inputs:
        - RGB Image (channels: 3 ; cols: 384 ; rows: 288) (Tensor 3 x 224 x 224)
        - Metadata (Tensor 9 x 1) (not used)
    Output:
        - Temperature (float)
    """

    def __init__(self):
        super(ResNet50_RgbNoMetadata, self).__init__()

        # Load a pre-trained ResNet50 network
        self.__resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Disable FineTuning on the complete model
        for param in self.__resnet.parameters():
            param.requires_grad = False

        # FineTune only the last ResNet layer
        for param in self.__resnet.layer4.parameters():
            param.requires_grad = True

        # Image pre-processing layer (resize, center cropping, ImageNet normalization)
        self.__resnet_preprocess = ResNet50_Weights.DEFAULT.transforms(antialias=False)

        # Modify the last FC layer to output a single value (instead of 1000 classes)
        self.__resnet.fc = Linear(2048, 1)

    def forward(self, image: Tensor, metadata: Tensor):
        # ResNet forward
        return self.__resnet.forward(self.__resnet_preprocess(image))


class ResNet50_RgbMetadata(Module):
    """
    Model based on a pre-trained ResNet50 architecture, using an RGB input image and its
    complete metadata.

    The complete ResNet50 model is frozen, except for the last convolution block. The
    last ResNet50 linear layer (2048 -> 1000) is removed. The 2048 ResNet50 output are
    concatenated with the metada (9) and pass through a linear layer (2057 -> 1) to
    output a unique continuous value (= temperature).

    Inputs:
        - RGB Image (channels: 3 ; cols: 384 ; rows: 288) (Tensor 3 x 224 x 224)
        - Metadata (Tensor 9 x 1)
    Output:
        - Temperature (float)
    """

    def __init__(self):
        super(ResNet50_RgbMetadata, self).__init__()

        ################################
        # ResNet50 Network Configuration
        ################################

        # Import a pre-trained ResNet50 network
        self.__resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Do not fine-tune the ResNet50 weights
        for param in self.__resnet.parameters():
            param.requires_grad = False

        # FineTune only the last ResNet layer
        for param in self.__resnet.layer4.parameters():
            param.requires_grad = True

        # Image pre-processing layer (resize, center cropping, ImageNet normalization)
        self.__resnet_preprocess = ResNet50_Weights.DEFAULT.transforms(antialias=False)

        # Replace the default ResNet Fully-Connected layer by Identity (to get the raw 2048 values)
        self.__resnet.fc = Identity()

        ##############################
        # Network Output with Metadata
        ##############################

        # The output of ResNet50 is 2048 + 9 metadata parameters
        self.__fc = Linear(2048 + 9, 1)

    def forward(self, image: Tensor, metadata: Tensor):
        # ResNet image processing
        image = self.__resnet_preprocess(image)
        resnet_out = self.__resnet.forward(image)

        # Network Out with ResNet Out and metadata
        out = cat((resnet_out, metadata), 1)
        # No activation function to output a continuous value
        out = self.__fc(out)

        return out


class ResNet50_RgbMetadataMlp(Module):
    """
    Model based on a pre-trained ResNet50 architecture, using an RGB input image and its
    complete metadata.

    The complete ResNet50 model is frozen, except for the last convolution block. The
    last ResNet50 linear layer (2048 -> 1000) is removed. The metadata tensor (9) is
    first processed by a linear layer (9 -> 9), and the resulting output concatenated
    with the ResNet50 output (2048). Finally, this tensor passes through a linear layer
    (2057 -> 1) to output a unique continuous value (= temperature).

    Inputs:
        - RGB Image (channels: 3 ; cols: 384 ; rows: 288) (Tensor 3 x 224 x 224)
        - Metadata (Tensor 9 x 1)
    Output:
        - Temperature (float)
    """

    def __init__(self):
        super(ResNet50_RgbMetadataMlp, self).__init__()

        ################################
        # ResNet50 Network Configuration
        ################################

        # Import a pre-trained ResNet50 network
        self.__resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Do not fine-tune the ResNet50 weights
        for param in self.__resnet.parameters():
            param.requires_grad = False

        # FineTune only the last ResNet layer
        for param in self.__resnet.layer4.parameters():
            param.requires_grad = True

        # Image pre-processing layer (resize, center cropping, ImageNet normalization)
        self.__resnet_preprocess = ResNet50_Weights.DEFAULT.transforms(antialias=False)

        # Replace the default ResNet Fully-Connected layer by Identity (to get the raw 2048 values)
        self.__resnet.fc = Identity()

        #####################################
        # Multi-Layer Perceptron for Metadata
        #####################################

        # Number of features to extract from the metadata
        self.__mlp_out_size = 9
        # Input metadata size = 9
        self.__mlp = Linear(9, self.__mlp_out_size)

        ###########################################
        # Network Output with ResNet features + MLP
        ###########################################

        # Output of ResNet50 (= 2048) + Output of MLP
        self.__fc = Linear(2048 + self.__mlp_out_size, 1)

    def forward(self, image: Tensor, metadata: Tensor):
        # ResNet image processing
        image = self.__resnet_preprocess(image)
        resnet_out = self.__resnet.forward(image)

        # MLP metadata processing
        mlp_out = tanh(self.__mlp(metadata))

        # Network Out with ResNet Out and metadata
        out = cat((resnet_out, mlp_out), 1)
        # No activation function to output a continuous value
        out = self.__fc(out)

        return out


class ResNet18_RgbNoMetadata(Module):
    """
    Model based on a pre-trained ResNet18 architecture, using only an RGB input image
    (the metadata is not used).

    The complete ResNet18 model is frozen, except for the last convolution block. The
    last ResNet18 linear layer (512 -> 1000) is replaced by a trainable linear layer
    (512 -> 1) to output a unique continuous value (= temperature).

    Inputs:
        - RGB Image (channels: 3 ; cols: 384 ; rows: 288) (Tensor 3 x 224 x 224)
        - Metadata (Tensor 9 x 1) (not used)
    Output:
        - Temperature (float)
    """

    def __init__(self):
        super(ResNet18_RgbNoMetadata, self).__init__()

        # Load a pre-trained ResNet18 network
        self.__resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Disable FineTuning on the complete model
        for param in self.__resnet.parameters():
            param.requires_grad = False

        # FineTune only the last ResNet layer
        for param in self.__resnet.layer4.parameters():
            param.requires_grad = True

        # Image pre-processing layer (resize, center cropping, ImageNet normalization)
        self.__resnet_preprocess = ResNet18_Weights.DEFAULT.transforms(antialias=False)

        # Modify the last FC layer to output a single value (instead of 1000 classes)
        self.__resnet.fc = Linear(512, 1)

    def forward(self, image: Tensor, metadata: Tensor):
        # ResNet forward
        return self.__resnet.forward(self.__resnet_preprocess(image))


class ResNet18_RgbMetadata(Module):
    """
    Model based on a pre-trained ResNet18 architecture, using an RGB input image and its
    complete metadata.

    The complete ResNet18 model is frozen, except for the last convolution block. The
    last ResNet50 linear layer (512 -> 1000) is removed. The 512 ResNet18 output are
    concatenated with the metada (9) and pass through a linear layer (521 -> 1) to
    output a unique continuous value (= temperature).

    Inputs:
        - RGB Image (channels: 3 ; cols: 384 ; rows: 288) (Tensor 3 x 224 x 224)
        - Metadata (Tensor 9 x 1)
    Output:
        - Temperature (float)
    """

    def __init__(self):
        super(ResNet18_RgbMetadata, self).__init__()

        ################################
        # ResNet18 Network Configuration
        ################################

        # Import a pre-trained ResNet18 network
        self.__resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Do not fine-tune the ResNet18 weights
        for param in self.__resnet.parameters():
            param.requires_grad = False

        # FineTune only the last ResNet layer
        for param in self.__resnet.layer4.parameters():
            param.requires_grad = True

        # Image pre-processing layer (resize, center cropping, ImageNet normalization)
        self.__resnet_preprocess = ResNet18_Weights.DEFAULT.transforms(antialias=False)

        # Replace the default ResNet Fully-Connected layer by Identity (to get the raw 2048 values)
        self.__resnet.fc = Identity()

        ##############################
        # Network Output with Metadata
        ##############################

        # The output of ResNet18 is 512 + 9 metadata parameters
        self.__fc = Linear(512 + 9, 1)

    def forward(self, image: Tensor, metadata: Tensor):
        # ResNet image processing
        image = self.__resnet_preprocess(image)
        resnet_out = self.__resnet.forward(image)

        # Network Out with ResNet Out and metadata
        out = cat((resnet_out, metadata), 1)
        # No activation function to output a continuous value
        out = self.__fc(out)

        return out


class ResNet18_RgbMetadataMlp(Module):
    """
    Model based on a pre-trained ResNet18 architecture, using an RGB input image and its
    complete metadata.

    The complete ResNet18 model is frozen, except for the last convolution block. The
    last ResNet50 linear layer (512 -> 1000) is removed. The metadata tensor (9) is
    first processed by a linear layer (9 -> 9), and the resulting output concatenated
    with the ResNet50 output (512). Finally, this tensor passes through a linear layer
    (521 -> 1) to output a unique continuous value (= temperature).

    Inputs:
        - RGB Image (channels: 3 ; cols: 384 ; rows: 288) (Tensor 3 x 224 x 224)
        - Metadata (Tensor 9 x 1)
    Output:
        - Temperature (float)
    """

    def __init__(self):
        super(ResNet18_RgbMetadataMlp, self).__init__()

        ################################
        # ResNet18 Network Configuration
        ################################

        # Import a pre-trained ResNet18 network
        self.__resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Do not fine-tune the ResNet18 weights
        for param in self.__resnet.parameters():
            param.requires_grad = False

        # FineTune only the last ResNet layer
        for param in self.__resnet.layer4.parameters():
            param.requires_grad = True

        # Image pre-processing layer (resize, center cropping, ImageNet normalization)
        self.__resnet_preprocess = ResNet18_Weights.DEFAULT.transforms(antialias=False)

        # Replace the default ResNet Fully-Connected layer by Identity (to get the raw 2048 values)
        self.__resnet.fc = Identity()

        #####################################
        # Multi-Layer Perceptron for Metadata
        #####################################

        # Number of features to extract from the metadata
        self.__mlp_out_size = 9
        # Input metadata size = 9
        self.__mlp = Linear(9, self.__mlp_out_size)

        ###########################################
        # Network Output with ResNet features + MLP
        ###########################################

        # Output of ResNet18 (= 512) + Output of MLP
        self.__fc = Linear(512 + self.__mlp_out_size, 1)

    def forward(self, image: Tensor, metadata: Tensor):
        # ResNet image processing
        image = self.__resnet_preprocess(image)
        resnet_out = self.__resnet.forward(image)

        # MLP metadata processing
        mlp_out = tanh(self.__mlp(metadata))

        # Network Out with ResNet Out and metadata
        out = cat((resnet_out, mlp_out), 1)
        # No activation function to output a continuous value
        out = self.__fc(out)

        return out
