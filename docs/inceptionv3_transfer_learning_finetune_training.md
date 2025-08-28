# inceptionv3_transfer_learning_finetune_training.py

## Description
This script implements **transfer learning and fine-tuning** using the InceptionV3 model for multi-class waste classification. The pre-trained InceptionV3 is used as a feature extractor, with custom layers added for classification.

## Key Steps
- Loads **InceptionV3** pretrained on ImageNet and freezes the base layers initially  
- Adds **GlobalAveragePooling**, **Dropout**, and dense layers for classification  
- Unfreezes layers from **layer 249 onward** for fine-tuning  
- Compiles the model with **Adam optimizer**, categorical crossentropy, and metrics including **precision** and **recall**  
- Trains using `train_gen` and validates using `val_gen`  
- Uses **callbacks**: early stopping, learning rate reduction, and model checkpointing  

## Expected Output
- Fine-tuned InceptionV3 model  
- Training/validation accuracy, precision, recall per epoch  
- Saved best fine-tuned model (`best_inception_finetuned.h5`)  

## Dependencies
- Python 3.x  
- tensorflow / keras  
- numpy  
- sklearn  

## Notes
- Requires preprocessed image data generators (`train_gen`, `val_gen`) and class weights (`class_weight_dict`)  
- Image size: 150x150, batch size: 32, epochs: 20 + 5 fine-tuning (adjustable)  
