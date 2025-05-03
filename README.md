# 🤘 Hand Gesture Recognition using CNN (Rock, Paper, Scissors)

This project implements a Convolutional Neural Network (CNN) in TensorFlow/Keras to classify hand gestures — **Rock**, **Paper**, and **Scissors** — based on image data. The dataset is organized into `train`, `validation`, and `test` directories under a folder called `Hand Gesture Data`.

## 📁 Dataset Structure


Hand Gesture Data/
├── train/
│   ├── rock/
│   ├── paper/
│   └── scissor/
├── validation/
│   ├── rock/
│   ├── paper/
│   └── scissor/
└── test/
    ├── rock/
    ├── paper/
    └── scissor/

- Training set: 1680 images  
- Validation set: 0 images *(empty, can be populated for better validation accuracy)*  
- Test set: 248 images  

## 🚀 Features

- Image preprocessing and data augmentation using ImageDataGenerator
- CNN model with Conv2D, MaxPooling, Dense, and Dropout layers
- Classification of 3 hand gestures: Rock, Paper, and Scissors
- Achieves ~96% accuracy on test set

## 🧠 Model Architecture

Input: (300, 300, 3)  
↓ Conv2D (32 filters) + ReLU + MaxPooling  
↓ Conv2D (64 filters) + ReLU + MaxPooling  
↓ Conv2D (128 filters) + ReLU + MaxPooling  
↓ Flatten  
↓ Dense (128 units) + ReLU + Dropout  
↓ Dense (3 units) + Softmax

## 📦 Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Pillow

Install using:

pip install tensorflow numpy pillow

## ▶️ How to Run

Ensure your folder structure is:

/Hand Gesture Data/
    /train/
    /validation/
    /test/

Then run your training script (e.g. train_model.py):

python train_model.py

## 🧪 Training Results

Epoch | Loss   | Accuracy  
------|--------|---------  
1     | 1.2768 | 55.22%  
2     | 0.3457 | 85.40%  
3     | 0.2184 | 91.86%  
4     | 0.1509 | 95.11%  
5     | 0.1419 | 93.76%  
6     | 0.1204 | 95.05%  
7     | 0.1702 | 93.82%  
8     | 0.1030 | 95.82%  
9     | 0.0975 | 96.56%  
10    | 0.0785 | 97.06%  

**Test Accuracy**: 96.37%

## 📷 Sample Output

Epoch 10/10  
52/52 [==============================] - 157s 3s/step - loss: 0.0785 - accuracy: 0.9706  
8/8 [==============================] - 8s 704ms/step - loss: 0.0867 - accuracy: 0.9637  
Test accuracy: 0.9637096524238586

## 📌 Notes

- The `validation` folder is empty in this project; add validation images to improve performance monitoring.
- You can extend the model by adding more gesture classes and tuning hyperparameters.

## 📬 Contact

For suggestions or collaborations, feel free to reach out or fork the repository.
