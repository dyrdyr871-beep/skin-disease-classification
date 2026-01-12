# Skin Disease Classification using CNN

## 1. Project Title
Skin Disease Classification using Convolutional Neural Networks (CNN)

---

## 2. Problem Statement
Skin diseases such as melanoma, basal cell carcinoma, actinic keratosis and vascular lesions require early and accurate diagnosis. Manual diagnosis by dermatologists can be time‑consuming and may vary based on expertise. This project aims to use deep learning techniques to automatically classify skin disease images.

---

## 3. Objective
The objective of this project is to develop a machine learning model using Convolutional Neural Networks (CNN) that can classify skin images into four disease categories:
- Actinic Keratosis (AK)
- Basal Cell Carcinoma (BCC)
- Melanoma
- Vascular Lesions (VL)

---

## 4. Dataset Description
The dataset consists of labeled skin lesion images divided into four classes. The images are organized into training and testing folders.

Folder Structure:

SkinDataset  
├── train  
│   ├── ak  
│   ├── bcc  
│   ├── melanoma  
│   └── vl  
└── test  
    ├── ak  
    ├── bcc  
    ├── melanoma  
    └── vl  

The dataset is stored in Google Drive and loaded into Google Colab for training and testing.

---

## 5. Methodology / Approach
1. Mount Google Drive in Google Colab  
2. Load dataset from Drive  
3. Preprocess images (resize, normalize, augmentation)  
4. Build CNN model using TensorFlow and Keras  
5. Train the model on training images  
6. Validate using test images 
7. Plot accuracy graph  
8. Save the trained model  
9. Test the model using new unseen images  

---

## 6. Tools and Technologies Used
- Python  
- Google Colab  
- TensorFlow  
- Keras  
- NumPy  
- Matplotlib  
- Google Drive  
- GitHub  

---

## 7. Steps to Run the Project
1. Open the Google Colab notebook  
2. Mount Google Drive  
3. Ensure dataset is present in Drive  
4. Run all code cells to train the model  
5. The accuracy graph will be generated  
6. The trained model will be saved  
7. Use Step‑10 to test any new image  

---

## 8. Results / Output
The CNN model was trained successfully and achieved classification results for four skin disease types.  
The training and validation accuracy graph is shown below.

![Accuracy Graph](accuracy_plot.png)

The model can predict the disease class of a new unseen skin image.

---

## 9. Trained Model

Download it here:  
MODEL link: https://drive.google.com/file/d/1qjCbxnXMO-EwFoq0ZDi1L0Cilqa0KTa2/view?usp=sharing

---

## 10. Files in this Repository
- Skin_Disease_Classification.ipynb  
- accuracy_plot.png  
- README.md  
- Model download link (Google Drive)

