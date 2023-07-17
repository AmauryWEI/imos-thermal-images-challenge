# Models

## Overview

This folder contains Pytorch models targeting Challenge #2.

## Faster R-CNN models

Faster R-CNN reference: 
[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://doi.org/10.48550/arXiv.1506.01497).

3 different Faster R-CNN models are proposed:

- `FasterRcnnResnet50FpnV2`: 

A Faster R-CNN model using a pre-trained ResNet50-FPN backbone (weights from 
[Benchmarking Detection Transfer Learning with Vision Transformers](https://doi.org/10.48550/arXiv.2111.11429)) 
on the [COCO dataset](https://cocodataset.org/). This model has 43'256'153 parameters in
total.

- `FasterRcnnMobileNetV3LargeFpn`

A Faster R-CNN model using a pre-trained MobileNetV3-FPN (Large) backbone (weights from 
[Searching for MobileNetV3](https://doi.org/10.48550/arXiv.1905.02244)) 
on the [COCO dataset](https://cocodataset.org/). This model has 18'930'229 parameters in
total.

- `FasterRcnnMobileNetV3Large320Fpn`

A Faster R-CNN model using a pre-trained MobileNetV3-FPN (Large) backbone for 
low-resolution images (weights from 
[Searching for MobileNetV3](https://doi.org/10.48550/arXiv.1905.02244)) 
on the [COCO dataset](https://cocodataset.org/). This model has 18'930'229 parameters in
total.

## RetinaNet model

RetinaNet reference: 
[Focal Loss for Dense Object Detection](https://doi.org/10.48550/arXiv.1708.02002)

1 RetinaNet model is proposed:

- `RetinaNetResnet50FpnV2`:

A RetinaNet model using a pre-trained Resnet50-FPN backbone (weights from 
[Benchmarking Detection Transfer Learning with Vision Transformers](https://doi.org/10.48550/arXiv.2111.11429)) 
on the [COCO dataset](https://cocodataset.org/). This model has 36'351'606 parameters in
total.
