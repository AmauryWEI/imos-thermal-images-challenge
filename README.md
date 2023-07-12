# imos-thermal-images-challenge

This repository contains documentation and source code to tackle two deep-learning
challenges.

## Challenges Descriptions

### Challenge 1

Using this
[Long-term Thermal Drift Dataset](https://www.kaggle.com/datasets/ivannikolov/longterm-thermal-drift-dataset)
to implement a deep-learning based method to predict the temperature in Celsius using
the thermal images (and metadata if needed). Use the code provided in the repo for data
loading, and feel free to use pytorch or tensorflow as you see fit.

### Challenge 2

Implement a pedestrian detection method based of the labelled data. Since YOLOv5 and
Faster R-CNN are included in the repo, try to use a different method.

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

### Quick Start Guide

1. Split the raw `Image_Dataset` into three separate training, validation, and testing
sets:

    ```bash
    python data_splitting/create_split_dataset.py ./dataset/LTD_Dataset/LTD_Dataset/Image_Dataset/metada_images.csv -o ./dataset/LTD_Dataset/LTD_Dataset/Split_Dataset/
    ```

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
