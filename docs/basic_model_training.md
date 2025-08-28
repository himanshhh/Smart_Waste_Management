# basic_model_training.py

## Description
This script defines, compiles, and trains a basic Convolutional Neural Network (CNN) for trash classification using TensorFlow/Keras. It uses image generators and class weights to handle data imbalance.

## Key Steps
- Builds a **Sequential CNN model** with convolution, pooling, and dense layers  
- Uses **ReLU** activations and softmax output for multi-class classification  
- Compiles model with **Adam optimizer** and categorical crossentropy loss  
- Trains model using `train_gen` and validates with `val_gen`  
- Applies computed **class weights** to handle imbalance  
- Prints model summary and training time  

## Expected Output
- Model architecture summary  
- Training and validation loss/accuracy per epoch  
- Total training time in seconds  

## Dependencies
- Python 3.x  
- tensorflow / keras  
- numpy  
- time (standard library)  

## Notes
- Requires preprocessed data generators (`train_gen`, `val_gen`) and class weights (`class_weight_dict`) from preprocessing script.  
- Default training runs for **10 epochs**.  
