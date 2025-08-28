# data_preprocessing.py

## Description
This script handles preprocessing of the trash classification dataset. It uploads and extracts the dataset, prepares train/validation/test splits, sets up image data generators with augmentation, and computes class weights to handle class imbalance.

## Usage
Run the script in Google Colab or locally to generate preprocessed datasets for model training:
```bash
python data_preprocessing.py
```

## Key Steps
- Uploads and extracts dataset ZIP file
- Reads and processes annotation CSV files
- Sets up train, validation, and test generators with image augmentation
- Computes class weights for handling imbalanced classes

## Dependencies
- Python 3.x
- pandas
- numpy
- matplotlib
- Pillow (PIL)
- scikit-learn
- tensorflow / keras
- torch / torchvision
