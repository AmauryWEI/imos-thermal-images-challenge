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

```bash
mv LTD\ Dataset LTD_Dataset/
mv LTD_Dataset/LTD\ Dataset LTD_Dataset/LTD_Dataset
```

2. Rename the `LTD_Dataset/LTD_Dataset/Video Clips` folder to `LTD_Dataset/LTD_Dataset/Data`

```bash
mv LTD_Dataset/LTD_Dataset/Video\ Clips LTD_Dataset/LTD_Dataset/Data
```

3. Move the `LTD_Dataset/LTD_Dataset/metatada.csv` file inside the `LTD_Dataset/LTD_Dataset/Data/` folder

```bash
mv LTD_Dataset/LTD_Dataset/metatada.csv LTD_Dataset/LTD_Dataset/Data/
```

4. Replace all the spaces by underscores in subfolders of 
`Data_Annotated_Subset_Object_Detectors`:

```bash
cd Data_Annotated_Subset_Object_Detectors/
# Rename testing set directories
mv testing/Apr\ Month testing/Apr_Month
mv testing/Aug\ Month testing/Aug_Month
mv testing/Jan\ Month testing/Jan_Month
# Rename training set directories
mv testing/Feb\ Day testing/Feb_Day
mv testing/Feb\ Month testing/Feb_Month
mv testing/Feb\ Week testing/Feb_Week
mv testing/Mar\ Week testing/Mar_Week
```

After extraction and renaming, the folder structure should be:

    .
    ├── dataset/
    |   ├── Data_Annotated_Subset_Object_Detectors/
    |   |   ├── testing/
    |   |   |   ├── Apr_Month/
    |   |   |   ├── Aug_Month/
    |   |   |   └── Jan_Month/
    |   |   ├── training/
    |   |   |   ├── Feb_Day/
    |   |   |   ├── Feb_Month/
    |   |   |   ├── Feb_Week/
    |   |   |   └── Mar_Week/
    |   |   └── validation/
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

### Challenge #1 

To generate the dataset of unique thermal images, run the following command:

    ```zsh
    pyenv shell imos    # Activate the virtual environment
    cd dataset/LTD_Dataset/LTD_Dataset
    python extract_images.py
    ```

Disclaimer: this command will take a while (10s of minutes). At the end of the dataset
generation, you should obtain the following folder structure:

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
    |   |       ├── Image_Dataset/
    |   |       |   ├── 20200514/
    |   |       |   |   ├── clip_0_1331/
    |   |       |   |   |   ├── image_0000.jpg
    |   |       |   |   |   ├── ...
    |   |       |   |   |   └── image_XXXX.jpg
    |   |       |   |   |── .../
    |   |       |   |   └── clip_n_HHmm/
    |   |       |   ├── ...
    |   |       |   ├── YYYMMDD/
    |   |       |   └── metadata_images.csv
    |   |       ├── extract_images.py
    |   |       └── load_video_metadata.py
    |   └── README.md
    └── README.md

### Challenge #2

The dataset is ready to use, no action required.
