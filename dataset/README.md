# dataset

This folder is meant to contain the
[Long-term Thermal Drift Dataset](https://www.kaggle.com/datasets/ivannikolov/longterm-thermal-drift-dataset).

## Download

1. Download the
[Long-term Thermal Drift Dataset](https://www.kaggle.com/datasets/ivannikolov/longterm-thermal-drift-dataset)
as a `.zip` archive on your machine.

## Extraction

1. Extract the `archive.zip` at the root of this folder (*i.e.* `<repo_root>/dataset/`).

## Renaming

1. Rename the `LTD Dataset` folders to `LTD_Dataset` (with an underscore)

2. Rename the `Video Clips` folder to `Data`

3. Move the `metatada.csv` file inside the `/Data` folder

After extraction and renaming, the folder structure should be:

    .
    ├── dataset/
    |   ├── Data_Annotated_Subset_Object_Detectors/
    |   ├── Data_Subset_Autoencoders_Anomaly_Detectors/
    |   ├── LTD_Dataset/
    |   |   └── LTD_Dataset/
    |   |       ├── Data/
    |   |       |   ├── 20200514/
    |   |       |   |   ├── clip_0_1331.mp4
    |   |       |   |   └── ...
    |   |       |   ├── ...
    |   |       |   └── metadata.csv
    |   |       ├── extract_images.py
    |   |       └── load_video_metadata.py
    |   └── README.md
    └── README.md

## Dataset Generation

This section covers how to generate the required datasets to use in both challenges.

### Thermal Images

To generate the dataset of unique thermal images, run the following command:

    ```zsh
    pyenv shell imos    # Activate the virtual environment
    python extract_images.py
    ```
