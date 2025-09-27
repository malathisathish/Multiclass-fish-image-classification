# 🐟 Multiclass Fish Image Classification
A comprehensive deep learning project for classifying fish images into multiple species using CNN architectures and transfer learning with pre-trained models.

## 📋 Table of Contents

◘ Project Overview

◘ Features

◘ Dataset

◘ Models

◘ Installation

◘ Usage

◘ Project Structure

◘ Results

◘ Deployment

◘ Configuration

◘ Monitoring and Evaluation

◘ Tech Stack

◘ Acknowledgments

◘ Conclusion

◘ Contact

◘ Author

## 🎯 Project Overview

This project implements a multiclass fish species classification system using state-of-the-art deep learning techniques. The system compares multiple CNN architectures and transfer learning approaches to achieve optimal performance.

## Skills & Technologies Used

**Deep Learning**: CNN architectures, Transfer Learning

**Framework**: TensorFlow/Keras

**Languages**: Python

**Deployment**: Streamlit

**Data Processing**: Data Augmentation, Preprocessing

**Evaluation**: Model Comparison, Metrics Analysis

**Visualization**: Matplotlib, Seaborn

## Business Use Cases

**Enhanced Accuracy**: Determine the best model architecture for fish image classification

**Deployment Ready**: Create a user-friendly web application for real-time predictions

**Model Comparison**: Evaluate and compare metrics across models to select the most suitable approach

## ✨ Features

**Multiple Model Architectures**: CNN from scratch + 5 pre-trained models

**Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score analysis

**Data Augmentation**: Rotation, zoom, flip, and other techniques

**Interactive Web App**: Streamlit-based deployment for real-time predictions

**Model Comparison**: Detailed performance comparison with visualizations

**Confusion Matrix Analysis**: Per-class performance evaluation

**Best Model Selection**: Automatic selection and saving of the best performing model

## 📊 Dataset

The dataset consists of fish images categorized into folders by species. 
**The system automatically handles**: 

- Data loading from zip files

- Train/validation splitting (80/20)

- Image preprocessing and normalization

- Data augmentation for training robustness

## Data Preprocessing Steps

**Image Rescaling**: Normalize pixel values to [0, 1] range

**Data Augmentation**: - Rotation (±30 degrees) 
- Width/Height shift (±20%) - Shear transformation
- - Zoom (±20%) - Horizontal flipping

**Validation Split**: 20% of data reserved for validation

## The fish classes in the dataset are:

Animal Fish

Animal Fish Bass

Fish Sea Food Black Sea Sprat

Fish Sea Food Gilt Head Bream

Fish Sea Food Horse Mackerel

Fish Sea Food Red Mullet

Fish Sea Food Red Sea Bream

Fish Sea Food Sea Bass

Fish Sea Food Shrimp

Fish Sea Food Striped Red Mullet

Fish Sea Food Trout

## 🤖 Models

1. ### CNN from Scratch

Custom architecture with 3 convolutional blocks - Dropout for regularization - Flatten and dense layers for classification

2. ### Transfer Learning Models

**VGG16**: Deep architecture with small filters

**ResNet50**: Skip connections for deeper networks

**MobileNet**: Lightweight architecture for efficiency

**InceptionV3**: Multi-scale feature extraction

**EfficientNetB0**: Compound scaling for optimal performance

## All transfer learning models use:

Pre-trained ImageNet weights

Frozen base layers

Custom classification head

Fine-tuning capabilities

## 🚀 Installation

#### Prerequisites
bash
Python 3.12.10

#### Required Libraries
bash
pip install tensorflow==2.17.0
pip install streamlit==1.39.0
pip install scikit-learn==1.5.2
pip install matplotlib==3.9.2
pip install seaborn==0.13.2
pip install pandas==2.2.2
pip install numpy==1.26.4
pip install Pillow==10.4.0
pip install joblib==1.4.2

### Clone Repository
bash
git clone https://github.com/Malathisathish/fish-classification.git
cd fish-classification
#### pip install -r requirements.txt

## 📖 Usage

### 1. Training Models
python
**In Google Colab**
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import files
import joblib

**Mount Google Drive**
from google.colab import drive
drive.mount('/content/drive')

**Unzip dataset**
!unzip /content/drive/MyDrive/Copy of Dataset.zip -d /content/fish_dataset

**Data loading and augmentation**
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('/content/fish_dataset/images.cv_jzk6llhf18tm3k0kyttxz/trin', target_size=(224,224), batch_size=32, class_mode='categorical')
val_generator = val_datagen.flow_from_directory('/content/fish_dataset/images.cv_jzk6llhf18tm3k0kyttxz/validation', target_size=(224,224), batch_size=32, class_mode='categorical')
test_generator = val_datagen.flow_from_directory('/content/fish_dataset/images.cv_jzk6llhf18tm3k0kyttxz/test', target_size=(224,224), batch_size=32, class_mode='categorical')
num_classes = len(train_generator.class_indices)
joblib.dump(train_generator.class_indices, 'class_names.pkl')
files.download('class_names.pkl')

**Train CNN from Scratch**
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
#### def build_cnn_scratch():
model = Sequential([
Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
MaxPooling2D(2,2),
Conv2D(64, (3,3), activation='relu'),
MaxPooling2D(2,2),
Conv2D(128, (3,3), activation='relu'),
MaxPooling2D(2,2),
Flatten(),
Dense(512, activation='relu'),
Dropout(0.5),
Dense(num_classes, activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
return model
cnn_model = build_cnn_scratch()
checkpoint = ModelCheckpoint('cnn_scratch_best.h5', monitor='val_accuracy', save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history_cnn = cnn_model.fit(
train_generator, epochs=50, validation_data=val_generator, callbacks=[checkpoint, early_stop]
)
**Train Transfer Learning Models**
from tensorflow.keras.applications import VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0
def build_transfer_model(base_model_name):
if base_model_name == 'VGG16':
base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
elif base_model_name == 'ResNet50':
base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
elif base_model_name == 'MobileNet':
base = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
elif base_model_name == 'InceptionV3':
base = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
elif base_model_name == 'EfficientNetB0':
base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base.trainable = False
model = Sequential([
base,
Flatten(),
Dense(512, activation='relu'),
Dropout(0.5),
Dense(num_classes, activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
base.trainable = True
fine_tune_at = len(base.layers) // 2
for layer in base.layers[:fine_tune_at]:
layer.trainable = False
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
return model
models = {}
histories = {}
pretrained_names = ['VGG16', 'ResNet50', 'MobileNet', 'InceptionV3', 'EfficientNetB0']
for name in pretrained_names:
model = build_transfer_model(name)
checkpoint = ModelCheckpoint(f'{name}_best.h5', monitor='val_accuracy', save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
train_generator, epochs=50, validation_data=val_generator, callbacks=[checkpoint, early_stop]
)
models[name] = model
histories[name] = history
histories['CNN_Scratch'] = history_cnn
**Step 6**: Evaluate and Save Best Model (use the code from previous responses)
After running, download best_fish_model.h5, class_names.pkl, df_comparison.csv.

### 2. Model Evaluation
After training, run the evaluation code in Colab to generate metrics:
python
df_comparison.to_csv('df_comparison.csv', index=False)
files.download('df_comparison.csv')

### 3. Streamlit Deployment
bash
streamlit run app.py
📁 Project Structure
├── Data
│   ├── train
│   ├── validation
│   ├── test
├── Project report.pdf
├── app.py
├── best_fish_model.h5
├── class_names.pkl
├── df_comparison.csv
├── requirements.txt
├── README.md


## 📈 Results
### Model Performance Comparison

| Model         | Accuracy | Precision | Recall | F1-Score | Inference Speed (ms) |
|---------------|---------|-----------|--------|----------|--------------------|
| VGG16         | 99.3%   | 99.4%     | 99.3%  | 99.3%    | 57.2               |
| ResNet50      | 88.3%   | 89.5%     | 88.3%  | 88.2%    | 97.8               |
| MobileNet     | 99.9%   | 99.9%     | 99.9%  | 99.9%    | 13.4               |
| InceptionV3   | 99.8%   | 99.8%     | 99.8%  | 99.8%    | 91.7               |
| EfficientNetB0| 89.2%   | 89.8%     | 89.2%  | 89.1%    | 20.1               |
| CNN Scratch   | 98.2%   | 98.1%     | 98.2%  | 98.1%    | 45.6               |


## Key Insights

**Best Overall Performance**: MobileNet achieves highest accuracy with reasonable model size 

- **Most Efficient**: MobileNet offers good performance with smallest model size

- **Custom CNN**: Competitive performance considering it's built from scratch

## 🌐 Deployment

#### Streamlit Web Application Features

**Image Upload**: Drag-and-drop or browse to upload fish images 

**Real-time Prediction**: Instant classification with confidence scores - Top-N 

**Predictions**: Display multiple possible classifications 

**Confidence Visualization**: Interactive charts showing prediction probabilities 

**Model Information**: Display model architecture and class information

## Deployment Steps

Ensure trained models are in the correct directory

Install required dependencies

Run the Streamlit application:
bash

streamlit run app.py

Access the web interface at http://localhost:8501

## Production Deployment Options

Streamlit Cloud: Direct deployment from GitHub

Heroku: Cloud platform deployment

AWS/GCP: Cloud service deployment

Docker: Containerized deployment

## 🔧 Configuration

#### Hyperparameters
python

#### Image dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224

#### Training parameters
BATCH_SIZE = 32
EPOCHS = 50 for cnn and 10 for pretrained model to reduce computation
LEARNING_RATE = 0.001

#### Data augmentation parameters
ROTATION_RANGE = 30
ZOOM_RANGE = 0.2
SHIFT_RANGE = 0.2

### Model Selection Criteria

**Accuracy**: Primary metric for model selection

**Model Size**: Consider deployment constraints

**Inference Speed**: Real-time prediction requirements

**Training Time**: Development efficiency

## 📊 Monitoring and Evaluation

#### Training Metrics

Training/Validation Accuracy

Training/Validation Loss

Learning Rate Schedule

Early Stopping Patience

## Evaluation Metrics

**Accuracy**: Overall classification accuracy

**Precision**: Per-class and weighted average

**Recall**: Per-class and weighted average

**F1-Score**: Harmonic mean of precision and recall

**Confusion Matrix**: Detailed per-class performance

## Visualization Features

Training history plots

Confusion matrices

Model comparison charts

Prediction probability distributions 

Performance radar charts

## 💻 Tech Stacks

🐍 Python → Core programming language for model development and data preprocessing.

🧠 TensorFlow / Keras → Building, training, and fine-tuning deep learning models (CNN, transfer learning).

📊 Pandas, NumPy → Data manipulation, preprocessing, and numerical computations.

📈 Matplotlib, Seaborn → Visualization of dataset distribution, training performance, and model evaluation.

🌐 Streamlit → Deployment of the model with an interactive user interface for fish image classification.

🔧 Git / GitHub → Version control and project collaboration.

🖼️ Pillow, Joblib → Image preprocessing and saving/loading trained models efficiently.

### 🙏 Acknowledgments

TensorFlow team for the excellent deep learning framework

Streamlit team for the intuitive web app framework

ImageNet dataset for pre-trained model weights

Open source community for various tools and libraries

Guvi mentors for supporting me throughout the project completion

## 🏁 Conclusion
This project successfully demonstrates a multiclass fish species classification system using CNN + transfer learning.

♠ MobileNet achieved the best accuracy (99.9%).

♠ Streamlit integration makes deployment easy and user-friendly.

♠ The solution can support fisheries, marine research, and ecological monitoring. 🌊🐟

## 🖋️ Author Details

**Name**: Malathi Y (Janani)

**Role**: Data Science Enthusiast | Former Staff Nurse

**Location**: Tamil Nadu, India

## About the Author:
Malathi is a passionate data scientist with experience in Python, SQL, and deep learning.
She loves building interactive dashboards and machine learning projects that solve real-world problems. 
With a unique background in healthcare, she combines analytical skills with a deep understanding of practical applications.

## Technical Skills of author:

**Programming & Data Analysis**: Python, Pandas, NumPy, SQL

**Machine Learning / Deep Learning**: CNN, Transfer Learning, TensorFlow/Keras, Scikit-learn

**Visualization & Dashboarding**: Streamlit, Plotly, Matplotlib, Seaborn

**Tools & Collaboration**: Git, GitHub, VS Code

## 📞 Contact
For questions, suggestions, or collaborations:

**Email**: malathisathish2228@gmail.com

**GitHub**: @https://github.com/malathisathish

**LinkedIn**: (www.linkedin.com/in/malathi-sathish-016a03354)

Built with ❤️ by malathi using TensorFlow and Streamlit

























































ModelAccuracyPrecisionRecallF1-ScoreSize (MB)VGG1699.3%99.4%99.3%99.3%57.2ResNet5088.3%89.5%88.3%88.2%97.8MobileNet99.9%99.9%99.9%99.9%13.4InceptionV399.8%99.8%99.8%99.8%91.7EfficientNetB089.2%89.8%89.2%89.1%20.1CNN Scratch98.2%98.1%98.2%98.1%45.6
Key Insights

Best Overall Performance: MobileNet achieves highest accuracy with reasonable model size - Most Efficient: MobileNet offers good performance with smallest model size - Custom CNN: Competitive performance considering it's built from scratch

🌐 Deployment
Streamlit Web Application Features

Image Upload: Drag-and-drop or browse to upload fish images - Real-time Prediction: Instant classification with confidence scores - Top-N Predictions: Display multiple possible classifications - Confidence Visualization: Interactive charts showing prediction probabilities - Model Information: Display model architecture and class information

Deployment Steps

Ensure trained models are in the correct directory
Install required dependencies
Run the Streamlit application:
bash
streamlit run app.py
Access the web interface at http://localhost:8501

Production Deployment Options

Streamlit Cloud: Direct deployment from GitHub
Heroku: Cloud platform deployment
AWS/GCP: Cloud service deployment
Docker: Containerized deployment

🔧 Configuration
Hyperparameters
python
Image dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224
Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
Data augmentation parameters
ROTATION_RANGE = 30
ZOOM_RANGE = 0.2
SHIFT_RANGE = 0.2
Model Selection Criteria

Accuracy: Primary metric for model selection
Model Size: Consider deployment constraints
Inference Speed: Real-time prediction requirements
Training Time: Development efficiency

📊 Monitoring and Evaluation
Training Metrics

Training/Validation Accuracy
Training/Validation Loss
Learning Rate Schedule
Early Stopping Patience

Evaluation Metrics

Accuracy: Overall classification accuracy
Precision: Per-class and weighted average
Recall: Per-class and weighted average
F1-Score: Harmonic mean of precision and recall
Confusion Matrix: Detailed per-class performance

Visualization Features

Training history plots
Confusion matrices
Model comparison charts
Prediction probability distributions - Performance radar charts

Development Guidelines

Follow PEP 8 style guidelines
Add comprehensive docstrings
Include unit tests for new features - Update documentation for any changes

💻 Tech Stacks

🐍 Python → Core programming language for model development and data preprocessing.
🧠 TensorFlow / Keras → Building, training, and fine-tuning deep learning models (CNN, transfer learning).
📊 Pandas, NumPy → Data manipulation, preprocessing, and numerical computations.
📈 Matplotlib, Seaborn → Visualization of dataset distribution, training performance, and model evaluation.
🌐 Streamlit → Deployment of the model with an interactive user interface for fish image classification.
🔧 Git / GitHub → Version control and project collaboration.
🖼️ Pillow, Joblib → Image preprocessing and saving/loading trained models efficiently.

🙏 Acknowledgments

TensorFlow team for the excellent deep learning framework
Streamlit team for the intuitive web app framework
ImageNet dataset for pre-trained model weights
Open source community for various tools and libraries
Guvi mentors for supporting me throughout the project completion

🏁 Conclusion
This project successfully demonstrates a multiclass fish species classification system using CNN + transfer learning.
♠ MobileNet achieved the best accuracy (99.9%).
♠ Streamlit integration makes deployment easy and user-friendly.
♠ The solution can support fisheries, marine research, and ecological monitoring. 🌊🐟
📞 Contact
For questions, suggestions, or collaborations:

Email: malathisathish2228@gmail.com
GitHub: @Malathisathish
LinkedIn: Malathisathis ---
Built with ❤️ by malathi using TensorFlow and Streamlit
