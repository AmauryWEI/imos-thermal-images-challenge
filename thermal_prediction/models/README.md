# Models

## Overview

This folder contains Pytorch models targeting Challenge #1. All the models are taking as
input a rescaled image (with pixel values between 0 and 1), as well as normalized
metadata (without the temperature obvisouly) as input.

## Sample Model

This is a sample model for testing purposes only.

## CNN Model

1 single CNN model is proposed:

- `CnnModel`:

A simple Convolutional Neural Network (CNN) with 5 convolutional layers to process the
image, and a single linear layer to generate the temperature prediction. The extracted
CNN features are concatenated with the metadata before passing through the linear layer.

The main idea behind this model is to design a very simple model, train it from
scratch, and check if it has enough learning capacity for the temperature prediction
task.

## ResNet Models

ResNet reference:
[Deep Residual Learning for Image Recognition](https://doi.org/10.48550/arXiv.1512.03385)

6 different models based on ResNet architectures are proposed:

- `ResNet50_RgbNoMetadata`:

A pre-trained ResNet50 model using only the image (not the metadata) to perform the
temperature prediction. The last linear layer in ResNet50 has been replaced to predict a
single continuous value (the temperature in degrees Celsius).

- `ResNet18_RgbNoMetadata`:

Similar to the `ResNet50_RgbNoMetadata` model, but with a ResNet18 architecture (18
layers instead of 50).

- `ResNet50_RgbMetadata`:

A model combining both the image and the image metadata. The image is processed by a
pre-trained ResNet50 network, where the last linear layer has been removed. The ResNet50
features are then concatenated with the normalized metadata, and pass through a final
linear layer predicting the temperature in degrees Celsius.

- `ResNet18_RgbMetadata`:

Similar to the `ResNet18_RgbMetadata` model, but with a ResNet18 architecture (18
layers instead of 50).

- `ResNet50_RgbMetadataMlp`:

A model combining both the image and the image metadata. The image is processed by a
pre-trained ResNet50 network, where the last linear layer has been removed. The metadata
is processed by a single linear layer. The ResNet50 features are then concatenated with
the output of the metadata linear layer, and pass through a final linear layer
predicting the temperature in degrees Celsius.

- `ResNet18_RgbMetadataMlp`:

Similar to the `ResNet50_RgbMetadataMlp` model, but with a ResNet18 architecture (18
layers instead of 50).

The main idea behind the ResNet models is to evaluate how pre-trained networks perform,
if the metadata is improving the accuracy of the model, and how deep the model should be
(50 vs. 18 layers).

## MobileNet Models

MobileNetV3 reference:
[Searching for MobileNetV3](https://doi.org/10.48550/arXiv.1905.02244)

3 different models based on MobileNetV3 (small) architectures are proposed:

- `MobileNetV3Small_RgbNoMetadata`:

A pre-trained MobileNetV3 (Small) model using only the image (not the metadata) to
perform the temperature prediction. The last linear layer in ResNet50 has been replaced
to predict a single continuous value (the temperature in degrees Celsius).

This model is similar to `ResNet50_RgbNoMetadata` or `ResNet18_RgbNoMetadata`.

- `MobileNetV3Small_RgbMetadata`:

A model combining both the image and the image metadata. The image is processed by a
pre-trained MobileNetV3 (Small) network, where the last linear layer has been removed.
The MobileNetV3 features are then concatenated with the normalized metadata, and pass
through a final linear layer predicting the temperature in degrees Celsius.

This model is similar to `ResNet50_RgbMetadata` or `ResNet18_RgbMetadata`.

- `MobileNetV3Small_RgbNetadataMlp`:

A model combining both the image and the image metadata. The image is processed by a
pre-trained MobileNetV3 (Small) network, where the last linear layer has been removed.
The metadata is processed by a single linear layer. The MobileNetV3 features are then
concatenated with the output of the metadata linear layer, and pass through a final
linear layer predicting the temperature in degrees Celsius.

This model is similar to `ResNet50_RgbMetadataMlp` or `ResNet18_RgbMetadataMlp`.

The main idea behind the MobileNetV3 models is to evaluate how a relatively small
network would compare to a much larger ResNet backbone (18 or 50 layers).

## Multi-Layer Perceptron (MLP) models

2 different MLP models are proposed:

- `MlpModel`:

A simple Multi-Layer Perceptron model with 3 linear layers which only processes the
image metadata.

- `MlpModelDateTime`:

A simple Multi-Layer Perceptron model with 3 linear layers which only processes the day
and hour the image was taken at.

The main idea behind those two models is to evaluate if the metadata can actually
support temperature predictions, and if so, the metadata fields with the most impact.
