# Model Evaluation

## Description
Evaluation of the trained deep learning models on the test dataset was performed. It includes plotting training/validation accuracy and loss, generating classification reports, calculating mean average precision (mAP), and visualizing confusion matrices. Fine-tuned models are also evaluated to compare performance improvements.

## Evaluation Steps
1. Plot training and validation accuracy over epochs.
2. Plot training and validation loss over epochs.
3. Generate predictions on the test dataset.
4. Calculate classification metrics including precision, recall, and F1-score for each class.
5. Compute the mean average precision (mAP) score.
6. Display the confusion matrix for visual analysis of class-level performance.
7. Repeat the evaluation for fine-tuned models to assess improvement.
8. **Prediction Confidence (for MobileNetV2 and InceptionV3)**: Evaluate a single sample image to check the predicted class along with the confidence score (0–1), demonstrating model certainty in its classification.

## Metrics
- **Accuracy**
- **Loss**
- **Precision, Recall, F1-score** per class
- **mAP (Mean Average Precision)**
- **Confusion Matrix**
- **Prediction Confidence** for selected images (MobileNetV2 & InceptionV3)

## Notes
- The evaluation procedure is identical for MobileNetV2, ResNet50, EfficientNetB0, and InceptionV3, except for the prediction confidence step which is specific to MobileNetV2 and InceptionV3.
- Fine-tuned models generally show improved class-level precision and mAP scores.
- Confusion matrices help identify classes that are commonly misclassified.
- Prediction confidence demonstrates the model's certainty in classifying unseen images and is useful for assessing trustworthiness of the output.

