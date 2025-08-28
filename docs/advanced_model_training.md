# advanced_model_training.py

## Description
This script defines, compiles, and trains an advanced multi-class CNN for trash classification using TensorFlow/Keras. It incorporates regularization, dropout, and callbacks to improve generalization and performance.

## Key Steps
- Builds a **deep CNN** with multiple convolutional blocks, max-pooling, dropout, and L2 regularization  
- Dense block for final classification with softmax activation  
- Compiles model with **Adam optimizer** and categorical crossentropy loss  
- Uses **learning rate scheduler**, **early stopping**, and **model checkpointing** callbacks  
- Trains model with `train_gen` and validates with `val_gen`  
- Applies **class weights** to handle class imbalance  

## Expected Output
- Model architecture summary  
- Training and validation loss/accuracy per epoch  
- Saved best model (`best_model.h5`)  
- Adaptive learning rate through training  

## Dependencies
- Python 3.x  
- tensorflow / keras  
- numpy  
- time (standard library)  

## Notes
- Requires preprocessed data generators (`train_gen`, `val_gen`) and class weights (`class_weight_dict`) from preprocessing script.  
- Default training runs for **50 epochs**, but early stopping may halt training sooner.  
