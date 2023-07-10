# File:         mlp_models.py
# Date:         2023/07/10
# Description:  Models based Multi-Layer Perceptron architectures

from torch import Tensor, tanh
from torch.nn import Module, Linear, Dropout


class MlpModel(Module):
    """
    Multi-Layer Perceptron (MLP) models which only processes the metadata (not the image)

    This model can be used to understand which metadata parameters have the most effect
    on the temperature.

    Input:  Image (cols: 384 ; rows: 288) (Tensor 288 x 384) + Metadata (Tensor 9 x 1)
    Output: Temperature (float)
    """

    def __init__(self):
        super(MlpModel, self).__init__()

        self.__fc1 = Linear(9, 9)
        self.__dropout1 = Dropout(0.2)

        self.__fc2 = Linear(9, 9)
        self.__dropout2 = Dropout(0.2)

        self.__fc3 = Linear(9, 9)
        self.__dropout3 = Dropout(0.2)

        # Output a temperature at the end
        self.__out = Linear(9, 1)

    def forward(self, image: Tensor, metadata: Tensor):
        meta = tanh(self.__fc1(metadata))
        meta = self.__dropout1(meta)

        meta = tanh(self.__fc2(meta))
        meta = self.__dropout2(meta)

        meta = tanh(self.__fc3(meta))
        meta = self.__dropout3(meta)

        return self.__out(meta)


class MlpModelDateTime(Module):
    """
    Multi-Layer Perceptron (MLP) models which only processes the "Day" and "Hour" metadata
    (not the image nor the rest of the metadata)

    This model can be used to validate the impact of the date and time on the temperature

    Input:  Image (cols: 384 ; rows: 288) (Tensor 288 x 384) + Metadata (Tensor 9 x 1)
    Output: Temperature (float)
    """

    def __init__(self):
        super(MlpModelDateTime, self).__init__()

        self.__fc1 = Linear(2, 2)
        self.__dropout1 = Dropout(0.2)

        self.__fc2 = Linear(2, 2)
        self.__dropout2 = Dropout(0.2)

        self.__fc3 = Linear(2, 2)
        self.__dropout3 = Dropout(0.2)

        # Output a temperature at the end
        self.__out = Linear(2, 1)

    def forward(self, image: Tensor, metadata: Tensor):
        # Column 7 = "Day" ; Column 8 = "Hour"
        day_hour = metadata[:, 7:9]
        meta = tanh(self.__fc1(day_hour))
        meta = self.__dropout1(meta)

        meta = tanh(self.__fc2(meta))
        meta = self.__dropout2(meta)

        meta = tanh(self.__fc3(meta))
        meta = self.__dropout3(meta)

        return self.__out(meta)
