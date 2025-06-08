# Pneumonia Classification using ResNet-50 (Transfer Learning)

## 📌 Objective
Fine-tune a ResNet-50 model to classify chest X-ray images into Pneumonia and Normal categories using the PneumoniaMNIST dataset.

## 📁 Dataset
- **Name**: PneumoniaMNIST
- **Source**: [Kaggle](https://www.kaggle.com/datasets/rijulshr/pneumoniamnist/data)

## 🛠️ Environment Setup (Windows + Anaconda)

```bash
# 1. Create virtual environment
conda create -n tf_env python=3.10

# 2. Activate environment
conda activate tf_env

# 3. Install required packages
pip install tensorflow==2.19.0
pip install ipykernel
python -m ipykernel install --user --name=tf_env --display-name "Python tf_env"
conda install matplotlib
conda install scikit-learn
