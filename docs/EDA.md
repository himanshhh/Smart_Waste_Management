# EDA.ipynb

## Description
This script performs exploratory data analysis (EDA) on the trash classification dataset. It visualizes class distributions, inspects image dimensions, and applies transformations/augmentations to preview sample images.

## Key Steps
- Uploads and extracts dataset ZIP file  
- Loads annotation CSVs and prints dataset overview  
- Visualizes **class distribution** using bar plots  
- Implements a custom PyTorch `Dataset` class  
- Displays batches of sample images (with and without augmentation)  
- Analyzes and plots **image size distributions** for train/valid/test sets  

## Expected Output
- Bar chart of class distribution  
- Grids of sample images with applied transformations  
- Histograms of image widths and heights for each dataset split  

## Dependencies
- Python 3.x  
- google-colab (optional for Colab uploads)  
- pandas  
- matplotlib  
- Pillow (PIL)  
- torch / torchvision  
- tensorflow / keras (for compatibility)  

## Notes
- Script is designed for **Google Colab** (uses `files.upload()`), but can be adapted to local environments.  
- Ensure dataset structure includes `train`, `valid`, `test` folders each with `_annotations.csv` and corresponding images.  
