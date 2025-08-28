# mobilenetv2_transfer_learning_finetune_training.py

## Description
This script implements **transfer learning and fine-tuning** using the MobileNetV2 model for multi-class waste classification. The pre-trained model is used as a feature extractor, and the top layers are fine-tuned to improve performance on the custom dataset.

## Key Steps
- Loads **MobileNetV2** pretrained on ImageNet and freezes the base layers  
- Adds **GlobalAveragePooling**, **Dropout**, and fully connected layers for classification  
- Unfreezes the **last 30 layers** of MobileNetV2 for fine-tuning  
- Compiles model with **Adam optimizer**, categorical crossentropy, and metrics including **precision** and **recall**  
- Uses **callbacks**: early stopping, learning rate reduction, and model checkpointing  
- Trains with `train_gen` and validates with `val_gen`  

## Expected Output
- Fine-tuned MobileNetV2 model  
- Training/validation accuracy, precision, recall per epoch  
- Saved best fine-tuned model (`best_transfer_model.h5`)  

## Dependencies
- Python 3.x  
- tensorflow / keras  
- numpy  
- sklearn  

## Notes
- Requires preprocessed image data generators (`train_gen`, `val_gen`) and class weights (`class_weight_dict`)  
- Image size: 150x150, batch size: 32, epochs: 20 (adjustable)  
