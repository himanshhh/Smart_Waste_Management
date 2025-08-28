from google.colab import files
import zipfile
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import torchvision

#upload the zip file
uploaded = files.upload()

#unzip the files
zip_path = "trash sorting.v2i.tensorflow.zip"
extract_path = "/content/trash_data"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extracted to:", extract_path)

#paths
base_dir = "/content/trash_data"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "valid")
test_dir = os.path.join(base_dir, "test")

IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

#load annotation CSV filea
train_df = pd.read_csv(os.path.join(train_dir, '_annotations.csv'))
val_df = pd.read_csv(os.path.join(val_dir, '_annotations.csv'))
test_df = pd.read_csv(os.path.join(test_dir, '_annotations.csv'))

#full path to images
train_df['filename'] = train_df['filename'].apply(lambda x: os.path.join(train_dir, x))
val_df['filename'] = val_df['filename'].apply(lambda x: os.path.join(val_dir, x))
test_df['filename'] = test_df['filename'].apply(lambda x: os.path.join(test_dir, x))

#convert class column to string
train_df['class'] = train_df['class'].astype(str)
val_df['class'] = val_df['class'].astype(str)
test_df['class'] = test_df['class'].astype(str)

#data generators
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=15)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='class',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filename',
    y_col='class',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_gen = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col='class',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

#get number of classes
num_classes = len(train_gen.class_indices)
print("Number of classes:", num_classes)

#calculate class weights for imbalance handling
y_train = train_df['class'].astype('category').cat.codes.to_numpy()
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weight_dict)