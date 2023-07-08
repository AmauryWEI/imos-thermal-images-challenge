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
3. Create a Python virtual environment and install the necessary packages

```bash
pyenv virtualenv 3.10.12 imos
pip install -r requirements.txt
```

## Q&A

### Cannot install `decord` on Apple Silicon macOS

[Pypi](https://pypi.org/project/decord/#files) does not provide a pre-compiled arm64
version of the package. You will have to: download the `decord` source code, apply one
modification (as of [v0.6.0](https://github.com/dmlc/decord/releases/tag/v0.6.0)),
compile it locally, then install it in the Python virtual environment.

1. Clone the `decord` repo locally and checkout v0.6.0:

    ```zsh
    git clone --recursive https://github.com/dmlc/decord
    ```

2. Install the dependencies:

    ```zsh
    xcode-select --install
    brew install cmake ffmpeg
    ```

3. Apply the following `git diff` modification:

    ```zsh
    diff --git a/src/video/ffmpeg/ffmpeg_common.h b/src/video/ffmpeg/ffmpeg_common.h
    index b0b973f..f0f7316 100644
    --- a/src/video/ffmpeg/ffmpeg_common.h
    +++ b/src/video/ffmpeg/ffmpeg_common.h
    @@ -21,6 +21,7 @@
    extern "C" {
    #endif
    #include <libavcodec/avcodec.h>
    +#include <libavcodec/bsf.h>
    #include <libavformat/avformat.h>
    #include <libavformat/avio.h>
    #include <libavfilter/avfilter.h>
    diff --git a/src/video/video_reader.cc b/src/video/video_reader.cc
    index af4858d..0c67566 100644
    --- a/src/video/video_reader.cc
    +++ b/src/video/video_reader.cc
    @@ -146,7 +146,7 @@ VideoReader::~VideoReader(){
    void VideoReader::SetVideoStream(int stream_nb) {
        if (!fmt_ctx_) return;
        AVCodec *dec;
    -    int st_nb = av_find_best_stream(fmt_ctx_.get(), AVMEDIA_TYPE_VIDEO, stream_nb, -1, &dec, 0);
    +    int st_nb = av_find_best_stream(fmt_ctx_.get(), AVMEDIA_TYPE_VIDEO, stream_nb, -1, (const AVCodec **)&dec, 0);
        // LOG(INFO) << "find best stream: " << st_nb;
        CHECK_GE(st_nb, 0) << "ERROR cannot find video stream with wanted index: " << stream_nb;
        // initialize the mem for codec context
    ```

4. Compile the project source code:

    ```zsh
    mkdir build
    cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Release
    make -j $(nproc --all)
    ```

5. Install the Python bindings:

    ```zsh
    cd python
    pyenv shell imos
    python setup.py build
    python setup.py install
    ```
