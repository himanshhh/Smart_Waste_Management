# efficientnetb0_transfer_learning_finetune_training.py

## Description
This script performs **transfer learning and fine-tuning** using the EfficientNetB0 model for multi-class waste classification. The pre-trained EfficientNetB0 is used as a feature extractor, with additional layers added for the custom classification task.

## Key Steps
- Loads **EfficientNetB0** pretrained on ImageNet and freezes the base layers initially  
- Adds **GlobalAveragePooling**, **Dropout**, and fully connected layers for classification  
- Unfreezes layers from **layer 100 onward** for fine-tuning  
- Compiles the model with **Adam optimizer**, categorical crossentropy, and metrics including **precision** and **recall**  
- Trains using `train_gen` and validates using `val_gen`  
- Uses **callbacks**: early stopping, learning rate reduction, and model checkpointing  

## Expected Output
- Fine-tuned EfficientNetB0 model  
- Training/validation accuracy, precision, recall per epoch  
- Saved best fine-tuned model (`best_efficientnet_finetuned.h5`)  

## Dependencies
- Python 3.x  
- tensorflow / keras  
- numpy  
- sklearn  

## Notes
- Requires preprocessed image data generators (`train_gen`, `val_gen`) and class weights (`class_weight_dict`)  
- Image size: 150x150, batch size: 32, epochs: 20 + 5 fine-tuning (adjustable)  
