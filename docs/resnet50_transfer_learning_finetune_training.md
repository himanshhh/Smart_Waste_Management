# resnet50_transfer_learning_finetune_training.py

## Description
This script performs **transfer learning and fine-tuning** using the ResNet50 model for multi-class waste classification. The pre-trained ResNet50 is used as a feature extractor, with additional layers added for the custom classification task.

## Key Steps
- Loads **ResNet50** pretrained on ImageNet and freezes the base layers initially  
- Adds **GlobalAveragePooling**, **Dropout**, and fully connected layers for classification  
- Unfreezes **top layers from layer 140** onward for fine-tuning  
- Compiles the model with **Adam optimizer**, categorical crossentropy, and metrics including **precision** and **recall**  
- Trains using `train_gen` and validates using `val_gen`  
- Uses **callbacks**: early stopping, learning rate reduction, and model checkpointing  

## Expected Output
- Fine-tuned ResNet50 model  
- Training/validation accuracy, precision, recall per epoch  
- Saved best fine-tuned model (`best_resnet_finetuned_model.h5`)  

## Dependencies
- Python 3.x  
- tensorflow / keras  
- numpy  
- sklearn  

## Notes
- Requires preprocessed image data generators (`train_gen`, `val_gen`) and class weights (`class_weight_dict`)  
- Image size: 150x150, batch size: 32, epochs: 20 + 10 fine-tuning (adjustable)  
