# File:         sample_model.py
# Date:         2023/07/08
# Description:  Sample CNN model for testing purposes

from torch import Tensor, relu, cat
from torch.nn import Module, Conv2d, BatchNorm2d, Linear, MaxPool2d


class SampleModel(Module):
    """
    Basic Convolutional Neural Network with 3x3 kernels and batch normalization for
    temperature prediction on the Long-term Thermal Drift Dataset.

    Input:  Image (Tensor 384 x 288) + Metadata (Tensor 7 x 1)
    Output: Temperature (float)
    """

    def __init__(self):
        super(SampleModel, self).__init__()

        self.__cnn_fc_output_size = 10
        self.__mlp_output_size = 7
        self.__out_fc_input_size = self.__cnn_fc_output_size + self.__mlp_output_size

        # CNN Components
        self.cnn_conv1 = Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.cnn_bn1 = BatchNorm2d(32)
        self.cnn_pool1 = MaxPool2d(kernel_size=2, stride=2)

        self.cnn_conv2 = Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.cnn_bn2 = BatchNorm2d(64)
        self.cnn_pool2 = MaxPool2d(kernel_size=2, stride=2)

        self.cnn_conv3 = Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.cnn_bn3 = BatchNorm2d(128)
        self.cnn_pool3 = MaxPool2d(kernel_size=2, stride=2)

        self.cnn_conv4 = Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.cnn_bn4 = BatchNorm2d(256)
        self.cnn_pool4 = MaxPool2d(kernel_size=2, stride=2)

        self.cnn_conv5 = Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.cnn_bn5 = BatchNorm2d(512)
        self.cnn_pool5 = MaxPool2d(kernel_size=2, stride=2)

        self.cnn_fc = Linear(512 * 12 * 9, self.__cnn_fc_output_size)

        # MLP Components
        self.mlp_fc = Linear(7, self.__mlp_output_size)

        # Single output = temperature
        self.out_fc = Linear(self.__out_fc_input_size, 1)

    def forward(self, image: Tensor, metadata: Tensor):
        # CNN Forward
        cnn_out = self.cnn_bn1(relu(self.cnn_conv1(image)))
        cnn_out = self.cnn_pool1(cnn_out)

        cnn_out = self.cnn_bn2(relu(self.cnn_conv2(cnn_out)))
        cnn_out = self.cnn_pool2(cnn_out)

        cnn_out = self.cnn_bn3(relu(self.cnn_conv3(cnn_out)))
        cnn_out = self.cnn_pool3(cnn_out)

        cnn_out = self.cnn_bn4(relu(self.cnn_conv4(cnn_out)))
        cnn_out = self.cnn_pool4(cnn_out)

        cnn_out = self.cnn_bn5(relu(self.cnn_conv5(cnn_out)))
        cnn_out = self.cnn_pool5(cnn_out)

        cnn_out = cnn_out.view(cnn_out.size(0), -1)
        print(cnn_out.shape)
        cnn_out = self.cnn_fc(cnn_out)

        # MLP Forward
        mlp_out = relu(self.mlp_fc(metadata))

        # Network Out with CNN Out and MLP Out
        out = cat((cnn_out, mlp_out))
        out = relu(self.out_fc(out))

        return out
