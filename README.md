# Pneumonia Classification using ResNet-50 (Transfer Learning)

This repository contains the code for an assignment focused on fine-tuning a ResNet-50 model to classify chest X-ray images as either 'Normal' or 'Pneumonia'. The project is implemented using PyTorch for the deep learning model and includes robust data handling, class imbalance mitigation, and overfitting prevention strategies.

## 📌 Objective
The primary objective is to develop a robust deep learning model for distinguishing pneumonia from normal chest X-rays, addressing challenges such as class imbalance and preventing overfitting, and reporting the model's performance using appropriate metrics.

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
```

## 📈 Install Required Libraries

```bash
# TensorFlow (specific version)
pip install tensorflow==2.19.0

# Jupyter kernel support
pip install ipykernel
python -m ipykernel install --user --name=tf_env --display-name "Python tf_env"

# Data visualization
conda install matplotlib

# Machine learning tools
conda install scikit-learn
```

## 🚀 How to Run
1. Open Jupyter Notebook:
```bash
jupyter notebook
```
2. Run the notebook RS_main.ipynb step-by-step. It includes data loading, training, and evaluation.

## 📈 Evaluation Metrics
- Accuracy

- Precision/Recall/F1-score

- Confusion Matrix

## 🔧 Hyperparameters
Parameter	| Value

Learning Rate | 0.001

Batch Size |	32

Epochs |	15


## ⚖️ Class Imbalance Handling
- Used stratified split and class weights in the loss function to handle imbalance.

## 🛡️ Overfitting Prevention
- Data augmentation

- Dropout

- Early stopping

## 📄 Deliverables
- RS_main.ipynb: Training and evaluation notebook

- requirements.txt: Package list

- README.md: Setup & instructions

