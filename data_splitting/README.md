# Data Splitting

## Overview

This folder contains a script to split the raw
[Long-term Thermal Drift Dataset](https://www.kaggle.com/datasets/ivannikolov/longterm-thermal-drift-dataset)
into separate training / validation / testing sets.

## Usage instructions

- **Display help message**

```bash
python create_split_dataset.py -h # Option -h prints the help message
```

- **Standard usage**

```bash
python create_split_dataset.py <path_to_metadata_images.csv>
python create_split_dataset.py <path_to_metadata_images.csv> -o <custom_output_dir>
```
