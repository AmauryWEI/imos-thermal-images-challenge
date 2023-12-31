# imos-thermal-images-challenge

![Static Badge](https://img.shields.io/badge/version-v1.0.0-red)
![Static Badge](https://img.shields.io/badge/Python-white?logo=python)
![Static Badge](https://img.shields.io/badge/Pytorch-white?logo=pytorch)

This repository contains documentation and source code to tackle two deep-learning
challenges on the [Long-term Thermal Drift Dataset](https://www.kaggle.com/datasets/ivannikolov/longterm-thermal-drift-dataset).

## Challenges Descriptions

### Challenge 1

Using the
[Long-term Thermal Drift Dataset](https://www.kaggle.com/datasets/ivannikolov/longterm-thermal-drift-dataset),
implement a deep-learning based method to predict the temperature in Celsius using
the thermal images (and metadata if needed).

### Challenge 2

Implement a pedestrian detection method based of the labelled data. Try methods
different from YOLOv5 and Faster R-CNN.

## Getting Started

### Requirements

1. Clone this repository
2. Download the
[Long-term Thermal Drift Dataset](https://www.kaggle.com/datasets/ivannikolov/longterm-thermal-drift-dataset)
by following the instructions under [`/dataset/README.md`](./dataset/README.md)
3. Create a Python virtual environment and install the necessary packages (using
[pyenv](https://github.com/pyenv/pyenv) as an example).

```bash
pyenv virtualenv 3.10.12 imos
pip install -r requirements.txt
```

Here are the exact versions used and tested for this project:

| Package       | Version  |
| ------------- | -------- |
| numpy         | 1.25.1   |
| pandas        | 2.0.3    |
| decord        | 0.6.0    |
| torch         | 2.0.1    |
| torchvision   | 0.15.2   |
| torchaudio    | 2.0.2    |
| opencv-python | 4.8.0.74 |
| tqdm          | 4.65.0   |
| scipy         | 1.11.1   |
| matplotlib    | 3.7.2    |
| pycocotools   | 2.0.6    |

### Challenge #1

To get started on challenge #1, check the dedicated
[`/thermal_prediction/README.md`](./thermal_prediction/README.md) file, which contains
instructions to visualize the dataset, train models, and test them.

### Challenge #2

To get started on challenge #2, check the dedicated
[`/pedestrian_detection/README.md`](./pedestrian_detection/README.md) file, which
contains instructions to visualize the dataset, train models, and test them.

## Q&A

### Cannot install `decord` on Apple Silicon macOS

[Pypi](https://pypi.org/project/decord/#files) does not provide a pre-compiled arm64
version of the package. You will have to use a fork of the `decord` package, named
[`eva-decord`](https://github.com/georgia-tech-db/eva-decord), which contains multiple
fixes enabling `decord` for Apple Silicon Macs. Follow the official instructions to
build and install it:

1. Clone the `eva-decord` repo locally:

    ```zsh
    git clone --recursive https://github.com/georgia-tech-db/eva-decord
    ```

2. Install the dependencies:

    ```zsh
    xcode-select --install
    brew install cmake ffmpeg
    ```

3. Compile the project source code:

    ```zsh
    mkdir build
    cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Release
    make -j $(nproc --all)
    ```

4. Copy the `libdecord.dylib` built at the previous step in `<eva-decord>/python/`:

    ```zsh
    cp build/libdecord.dylib ./python/
    ```

5. Install the Python bindings:

    ```zsh
    cd python
    pyenv shell imos
    python setup.py build
    python setup.py install
    ```

### Facig a "Too many open files" exception

The root cause behind this exception is not clear. With two separate machines
running Ubuntu 20.04.6 LTS, one suffers from this issue and the other does not.
This problem is discussed in 
the [GitHub issue #11201](https://github.com/pytorch/pytorch/issues/11201) on 
the `pytorch` repository.

To mitigate the issue, you can add the following line at the beginning of the 
`main()` function in the `train.py` scripts:

```python
torch.multiprocessing.set_sharing_strategy('file_system')
```

### Facing a "NaN loss during training" exception

This issue might happen when training the Faster R-CNN MobileNetV3 networks with
multiple trainable layers, a small batch size, and a large learning rate. Try to
reduce the number of trainable layers (between 0 and 2), use a larger batch size
(*e.g.* 4, 8, 16, ...), and reduce the learning rate (*e.g.* 4e-5).
