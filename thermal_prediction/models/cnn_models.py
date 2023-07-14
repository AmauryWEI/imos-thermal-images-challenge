# File:         cnn_models.py
# Date:         2023/07/09
# Description:  Models based on convolutional architectures (CNNs)

from torch import Tensor, relu, cat
from torch.nn import Module, Conv2d, BatchNorm2d, Linear, MaxPool2d, Dropout


class CnnModel(Module):
    """
    Convolutional Neural Network with (5x5) and (3x3) kernels and 2D max pooling for
    temperature prediction on the Long-term Thermal Drift Dataset.

    During training, the network internally uses batch normalization and dropout (with a
    drop probability of 0.2).

    Input:  Image (cols: 384 ; rows: 288) (Tensor 288 x 384) + Metadata (Tensor 9 x 1)
    Output: Temperature (float)
    """

    def __init__(self):
        super(CnnModel, self).__init__()

        self.__cnn_fc_output_size = 50
        self.__mlp_output_size = 9
        self.__out_fc_input_size = self.__cnn_fc_output_size + self.__mlp_output_size

        self.dropout = Dropout(0.2)

        ################
        # CNN Components
        # Conv2d Out = 1 + (In + 2 * padding - dilation x (kernel_size - 1) - 1) / stride
        # MaxPool2d Out = 1 + (In + 2 * padding - dilation * (kernel_size - 1) - 1) / stride
        ################
        # Size = 288 x 384
        self.cnn_conv1 = Conv2d(1, 16, kernel_size=5, stride=1, padding=0)
        self.cnn_bn1 = BatchNorm2d(16)
        self.cnn_pool1 = MaxPool2d(kernel_size=4, stride=2)

        self.cnn_conv2 = Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.cnn_bn2 = BatchNorm2d(32)
        self.cnn_pool2 = MaxPool2d(kernel_size=4, stride=2)

        self.cnn_conv3 = Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.cnn_bn3 = BatchNorm2d(64)
        self.cnn_pool3 = MaxPool2d(kernel_size=4, stride=2)

        self.cnn_conv4 = Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.cnn_bn4 = BatchNorm2d(128)
        self.cnn_pool4 = MaxPool2d(kernel_size=4, stride=2)

        self.cnn_conv5 = Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.cnn_bn5 = BatchNorm2d(256)
        self.cnn_pool5 = MaxPool2d(kernel_size=4, stride=2)

        self.cnn_fc = Linear(256 * 5 * 8, self.__cnn_fc_output_size)

        ################
        # MLP Components
        ################
        self.mlp_fc = Linear(9, self.__mlp_output_size)

        ################
        # Single output = temperature
        ################
        self.out_fc = Linear(self.__out_fc_input_size, 1)

    def forward(self, image: Tensor, metadata: Tensor):
        # CNN Forward
        cnn_out = self.cnn_bn1(relu(self.cnn_conv1(image)))
        cnn_out = self.cnn_pool1(cnn_out)
        cnn_out = self.dropout(cnn_out)

        cnn_out = self.cnn_bn2(relu(self.cnn_conv2(cnn_out)))
        cnn_out = self.cnn_pool2(cnn_out)
        cnn_out = self.dropout(cnn_out)

        cnn_out = self.cnn_bn3(relu(self.cnn_conv3(cnn_out)))
        cnn_out = self.cnn_pool3(cnn_out)
        cnn_out = self.dropout(cnn_out)

        cnn_out = self.cnn_bn4(relu(self.cnn_conv4(cnn_out)))
        cnn_out = self.cnn_pool4(cnn_out)
        cnn_out = self.dropout(cnn_out)

        cnn_out = self.cnn_bn5(relu(self.cnn_conv5(cnn_out)))
        cnn_out = self.cnn_pool5(cnn_out)
        cnn_out = self.dropout(cnn_out)

        cnn_out = cnn_out.view(cnn_out.size(0), -1)
        cnn_out = self.cnn_fc(cnn_out)
        cnn_out = self.dropout(cnn_out)

        # MLP Forward
        mlp_out = relu(self.mlp_fc(metadata))

        # Network Out with CNN Out and MLP Out
        out = cat((cnn_out, mlp_out), 1)
        # No activation function to output a continuous value
        out = self.out_fc(out)

        return out
