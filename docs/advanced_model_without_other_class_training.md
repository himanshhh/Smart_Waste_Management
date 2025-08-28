# advanced_model_without_other_class_training.py

## Description
This script builds, compiles, and trains an advanced CNN for trash classification after removing the minority `'Other'` class from the dataset. The model uses regularization, dropout, and callbacks for improved generalization.

## Key Steps
- Determines number of classes after dropping `'Other'`  
- Constructs a **deep CNN** with multiple convolutional layers, max-pooling, and dropout  
- Dense block for multi-class softmax classification  
- Compiles with **Adam optimizer** and categorical crossentropy loss  
- Uses **early stopping**, **model checkpointing**, and **learning rate scheduling** callbacks  
- Trains with `train_gen` and validates with `val_gen`  

## Expected Output
- Model summary with updated number of classes  
- Training/validation loss and accuracy per epoch  
- Saved best model (`best_model.h5`)  

## Dependencies
- Python 3.x  
- tensorflow / keras  
- numpy  
- time (standard library)  

## Notes
- Requires preprocessed data generators (`train_gen`, `val_gen`) without the `'Other'` class.  
- Training defaults to **10 epochs**, adjustable as needed.  
