# 🦋 Butterfly Species Classification Using Transfer Learning

This project is a deep learning web application built using **Flask** that classifies butterfly species from images using **transfer learning** with models like **VGG16**, **EfficientNetB0**, and **ResNet50**.

## 📌 Project Overview

Butterflies play a crucial role in the ecosystem as pollinators and bio-indicators. Classifying butterfly species manually can be time-consuming and error-prone. This project automates the classification process using state-of-the-art CNN architectures and is suitable for biodiversity research and environmental monitoring.

## 🧠 Model Summary

Three transfer learning models were implemented:
- **VGG16**
- **EfficientNetB0** ✅ *Best performing*
- **ResNet50**

All models use:
- Pre-trained base (ImageNet)
- Global Average Pooling
- Dense layers with Dropout
- Softmax classifier

## 🧪 Dataset

- Images of multiple butterfly species stored in class-wise folders
- Images resized to `128x128`
- Normalized pixel values between `0–1`
- Train-validation split: 80% - 20%
- Augmented using Keras `ImageDataGenerator` (rotation, zoom, flips, etc.)

## ⚙️ Flask App 

Steps to run this app

1. pip install -r requirements.txt

2. python app.py
