# Weekly Journal 

Date: Week 28/07/2025 to 03/08/25 \
Location: Work from home \
Contributor: Himansh Arora
________________________________________
Progress Update:
For this week, I worked on 2 things:

1. Best Model
- Improving the performance of the best model obtained through cross-validation and hyperparameter tuning. 
- This was done by making changes to the preprocessing steps. 
- Sometimes excessive preprocessing can distort the image resolution and result in poor classification and training.
- For this, I removed the brightness and contrast settings of the preprocessing steps. This improved the model performance. 

2. Transfer Learning
- Now that I have established my best model, I try to introduce transfer Learning using pre-trained models. 
- The models for this purpose were chosen after performing appropriate literature review.
- This week, I tried transfer Learning on the MobileNetV2 model. 
- I also applied fine-tuning and compared the results of transfer Learning alone and with fine-tuning. 
- Applying fine-tuning and transfer Learning gave exceptionally good results for this dataset. 

Challenges Faced: 
- It was difficult to identify which preprocessing steps to after to improve the model performance and after hit and trial I was able to make a decision. 
- The best model still showed confusion between classes.
- This confusion was further overcome by introducing transfer Learning with fine-tuning. 

Next Steps: Fit more pre-trained models using Transfer Learning and fine-tuning for our given task of waste classification.  The results will be compared based on MAP, precision, recall and accuracy.
________________________________________
Next Journal: \
Date: 10/08/2025 \
Journal Prepared by: Himansh Arora \