# Masters Dissertation  
**ScrapStrat - Smart Waste Management with Deep Learning**

## Project Description
This project explores the application of deep learning for **waste image classification** to aid sustainable waste management. The goal is to accurately identify different types of waste items and classify them into appropriate categories, which can help users dispose of waste efficiently and responsibly.  

The dataset used contains **2,348 annotated images** categorized into Glass, Plastic, Metal, Bin, and Other. To address data imbalance and variability, the project applies extensive preprocessing and augmentation techniques. Multiple deep learning approaches are explored, including models built from scratch and **transfer learning with fine-tuned architectures**.  

The three best models identified in this study are:  
- A custom CNN (after hyperparameter tuning and stratified k-fold cross-validation)  
- MobileNetV2 (transfer learning + fine-tuning)  
- InceptionV3 (transfer learning + fine-tuning)  

Among these, **MobileNetV2 achieved the best overall performance** with a mean average precision (mAP) of **0.76**, followed by InceptionV3 (mAP 0.72) and the tuned CNN from scratch (mAP 0.68).  

This project demonstrates how preprocessing, cross-validation, and model selection significantly improve waste classification performance, with potential for real-world deployment in smart waste management systems.

---

## Dataset
- Source: [Roboflow Trash Sorting Dataset](https://universe.roboflow.com/jawads-workspace/trash-sorting-037nw/dataset/2)  
- Total Images: 2,348  
- Categories: Glass, Plastic, Metal, Bin, Other  
- Preprocessing: resizing, normalization, rotation, flipping, and removal of minority `Other` class in certain experiments.  

---

## Technologies Used
- **Languages & Frameworks:** Python, TensorFlow, Keras, PyTorch  
- **Libraries:** NumPy, Pandas, OpenCV, Matplotlib, Seaborn  
- **Techniques:** Convolutional Neural Networks (CNN), Hyperparameter Tuning, Stratified K-Fold Cross-Validation, Transfer Learning (MobileNetV2, InceptionV3, EfficientNetB0, ResNet50), Fine-Tuning  

---

## Methodology
1. **Data Analysis & Preprocessing**  
   - Standardizing image sizes  
   - Checking annotation consistency  
   - Image augmentations (rotation, flipping, brightness/contrast adjustments)  
   - Handling class imbalance with weighting and stratified k-fold  

2. **Model Development**  
   - Basic CNN from scratch  
   - Advanced CNN with regularization and dropout  
   - Hyperparameter tuning (batch size, dropout, epochs)  
   - Stratified k-fold cross-validation  

3. **Transfer Learning & Fine-Tuning**  
   - MobileNetV2  
   - InceptionV3  
   - EfficientNetB0  
   - ResNet50  

4. **Evaluation Metrics**  
   - Accuracy, Precision, Recall, F1-Score  
   - Mean Average Precision (mAP)  
   - Confusion Matrix  
   - Single-image testing with prediction confidence  

---

## Results
- Best CNN from scratch: **Accuracy ~0.65, mAP = 0.68**  
- MobileNetV2 (fine-tuned): **mAP = 0.76**, best overall performance  
- InceptionV3 (fine-tuned): **mAP = 0.72**, highest single-image confidence (100%)  
- EfficientNetB0 & ResNet50 underperformed relative to the above models  

Final takeaway: **MobileNetV2 provides the best trade-off between accuracy, confidence, and training efficiency.**

---

## Future Work
- Further hyperparameter tuning (e.g., learning rate optimization)  
- Exploration of additional transfer learning architectures  
- Deployment of a **real-time waste classification application with user interface**  
- Classification into **recyclable, non-recyclable, and compostable** categories  
- Business model exploration for scalable smart waste management  

---

## Installation
Clone this repository and install required dependencies:  
```bash
git clone <https://gitlab.comp.dkit.ie/D00233455/masters-dissertation>
cd smart-waste-management
pip install -r requirements.txt
```

---

## Acknowledgments
- Dataset: Roboflow 
- Deep learning architectures: TensorFlow / Keras applications
- Supervised by: Dr Peadar Grant and Dr Siobhan Connolly Kernan

---

## Author
- Himansh Arora
- Master's in Data Analytics
- [LinkedIn](https://www.linkedin.com/in/himansh-arora-a321471a1/)
- [Github](https://github.com/himanshhh)