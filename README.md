# 🧠 Pneumonia Detection Using ResNet-50

This project leverages **transfer learning** on a pre-trained **ResNet-50** model to classify chest X-ray images into **Pneumonia** or **Normal** classes. The dataset used is **PneumoniaMNIST**, and this project was built for a medical imaging assignment.

---

## 📌 Objective

- Fine-tune a ResNet-50 model to classify X-ray images.
- Address class imbalance using smart techniques.
- Evaluate performance using well-justified metrics.
- Apply regularization to prevent overfitting.

---

## 📁 Dataset

- **Name**: PneumoniaMNIST  
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/rijulshr/pneumoniamnist/data)  
- **Format**: Images grouped into class folders

> 📥 **Important**: After downloading, unzip the dataset and place it inside a folder named `data/` at the root of this project.

---

## 🔧 Environment Setup

You can set up the project environment in two ways: using Conda or manually with pip.

---

### 🧪 Option 1: Conda Environment via `env.yaml`

Create the environment using the provided YAML file:

```bash
conda env create -f env.yaml
conda activate tf_env
If env.yaml is not provided, see Option 2 to set it up manually.

🧪 Option 2: Manual Setup Using Anaconda Prompt (Windows)
bash
Copy
Edit
# Step 1: Create environment
conda create -n tf_env python=3.10

# Step 2: Activate environment
conda activate tf_env

# Step 3: Install dependencies
pip install tensorflow==2.19.0
pip install ipykernel
python -m ipykernel install --user --name=tf_env --display-name "Python (tf_env)"

# Step 4: Visualization and ML tools
conda install matplotlib
conda install scikit-learn
📦 Required Dependencies
Also available in requirements.txt:

txt
Copy
Edit
python==3.10
tensorflow==2.19.0
ipykernel
matplotlib
scikit-learn
To install all dependencies from the file:

bash
Copy
Edit
pip install -r requirements.txt
🚀 How to Run the Project
Launch Jupyter Notebook:

bash
Copy
Edit
jupyter notebook
Open and execute all cells in the notebook:

text
Copy
Edit
RS_main.ipynb
Make sure the dataset is placed in the ./data/ directory before you start.

⚙️ Model Details
Feature	Details
Base Model	ResNet-50 (ImageNet pretrained)
Input Size	224x224 RGB
Output Layer	Dense (sigmoid for binary)
Loss Function	Binary Cross-Entropy
Optimizer	Adam
Framework	TensorFlow / Keras

🎛️ Hyperparameters
Parameter	Value
Learning Rate	0.001
Batch Size	32
Epochs	15
Optimizer	Adam

📈 Evaluation Metrics
We used 3 main metrics for robust evaluation:

Accuracy - For overall correctness

F1-Score - Balanced metric (precision + recall) for imbalanced data

Confusion Matrix - To inspect false positives/negatives in detail

⚖️ Handling Class Imbalance
To address class imbalance (e.g., fewer Pneumonia cases):

Used class_weight during training (computed from label distribution)

Ensures that minority class gets appropriate attention

🛡️ Overfitting Prevention Techniques
Implemented multiple regularization strategies:

Data Augmentation: Image rotation, flipping, zoom, shift

Dropout Layers: Added in fully connected layers

Early Stopping: Stops training when validation loss plateaus

Validation Split: Helps detect overfitting early

🧪 Reproducibility Checklist
✅ Conda-based environment OR pip-based setup

✅ Dataset stored locally in ./data/

✅ Jupyter notebook with reproducible steps

✅ Metrics and hyperparameters documented

📂 Project Structure
bash
Copy
Edit
.
├── RS_main.ipynb           # Jupyter notebook for training and evaluation
├── data/                   # Dataset folder (manually added)
├── requirements.txt        # Required libraries
├── env.yaml                # (Optional) Conda environment file
└── README.md               # This file
👤 Author
Developed by [Your Name] as part of a deep learning assignment focused on medical image classification.

📜 License
This project is intended for academic and educational use only. Refer to the original dataset's license for reuse permissions.

yaml
Copy
Edit

---
