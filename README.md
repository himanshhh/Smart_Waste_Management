# ♻️ ScrapStrat - Smart Waste Management with Deep Learning

This **Master's Dissertation** explores the application of deep learning for **waste image classification** to aid sustainable waste management. The goal is to accurately identify different types of waste items and classify them into appropriate categories, which can help users dispose of waste efficiently and responsibly.

---

## 📌 Objective

- Perform **image classification** of waste items using **CNN** models built from scratch and **transfer learning**.  
- Apply **data augmentation**, **hyperparameter tuning**, and **stratified k-fold cross-validation** to improve model performance.  
- Compare and fine-tune models such as **MobileNetV2**, **InceptionV3**, **ResNet50**, and **EfficientNetB0** for optimal classification.  
- Support **smart and sustainable waste management** practices.  

---

## 🛠 Technologies & Tools Used

| Category | Tools / Libraries |
|----------|------------------|
| Languages & Frameworks | Python, TensorFlow, Keras, PyTorch |
| Libraries | NumPy, Pandas, OpenCV, Matplotlib, Seaborn |
| Techniques | CNN, Transfer Learning, Fine-Tuning, Hyperparameter Tuning, Stratified K-Fold Cross-Validation, Data Augmentation |
| Evaluation | Accuracy, Precision, Recall, F1-Score, Mean Average Precision (mAP) |

---

## 📁 Dataset Overview

- **Source:** [Roboflow Trash Sorting Dataset](https://universe.roboflow.com/jawads-workspace/trash-sorting-037nw/dataset/2)  
- **Total Images:** 2,348  
- **Categories:** Glass, Plastic, Metal, Bin, Other  
- **Preprocessing & Augmentation:** Resizing, normalization, rotation, flipping, brightness/contrast adjustment, minority class removal  

---

## 🔍 Methodology

### ✅ Data Analysis & Preprocessing
- Standardized image sizes and checked annotation consistency  
- Applied **image augmentations** to increase dataset variability  
- Addressed class imbalance using weighting and stratified k-fold cross-validation  

### 🧠 Model Development
- Built **basic and advanced CNNs** with regularization and dropout  
- Applied **hyperparameter tuning** (batch size, dropout, epochs)  
- Evaluated models using **cross-validation**  

### 🌐 Transfer Learning & Fine-Tuning
- Fine-tuned **MobileNetV2**, **InceptionV3**, **ResNet50**, and **EfficientNetB0**  
- Tested top layers on single images to assess prediction confidence  

---

## 📊 Results

- **Custom CNN:** Accuracy ~0.65, mAP = 0.68  
- **MobileNetV2 (fine-tuned):** mAP = 0.76, best overall performance  
- **InceptionV3 (fine-tuned):** mAP = 0.72, highest single-image confidence  
- EfficientNetB0 & ResNet50 underperformed relative to top models  

**Key takeaway:** MobileNetV2 provides the best trade-off between **accuracy, confidence, and efficiency**.

---

## 🔮 Future Work

- Further **hyperparameter optimization** (learning rate, batch size)  
- Testing additional **transfer learning architectures**  
- Deploying a **real-time waste classification application** with a user interface  
- Categorization into **recyclable, non-recyclable, and compostable waste**  
- Exploring scalable **smart waste management solutions**

---

## 🏆 Acknowledgments

- Dataset: Roboflow  
- Deep Learning Architectures: TensorFlow / Keras  
- Supervision: Dr Peadar Grant & Dr Siobhan Connolly Kernan  

---

## 📄 Full Dissertation

The complete dissertation detailing methodology, experiments, and results is available as a PDF in this repository:  
[Download Thesis PDF](./theis.pdf)

---

## ⚙️ Installation
Clone this repository and install required dependencies:  
```bash
git clone <https://gitlab.comp.dkit.ie/D00233455/masters-dissertation>
cd smart-waste-management
pip install -r requirements.txt
```

---

## ✍️ Author

**Himansh Arora**  
Master's in Data Analytics  
[LinkedIn](https://www.linkedin.com/in/himansh-arora-a321471a1/)
