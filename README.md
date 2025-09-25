# 🐟 Multiclass Fish Image Classification 
A comprehensive deep learning project for classifying fish images into multiple species using CNN architectures and transfer learning with pre-trained models.

## 📋 Table of Contents 
- [Project Overview](#project-overview) - [Features](#features) - [Dataset](#dataset) - [Models](#models) - [Installation](#installation) - [Usage](#usage) - [Project Structure](#project-structure) - [Results](#results) - [Deployment](#deployment) 

## 🎯 Project Overview 
This project implements a multiclass fish species classification system using state-of-the-art deep learning techniques. The system compares multiple CNN architectures and transfer learning approaches to achieve optimal performance. 

### Skills & Technologies Used 
- **Deep Learning**: CNN architectures, Transfer Learning - **Framework**: TensorFlow/Keras - **Languages**: Python - **Deployment**: Streamlit - **Data Processing**: Data Augmentation, Preprocessing - **Evaluation**: Model Comparison, Metrics Analysis - **Visualization**: Matplotlib, Seaborn 

### Business Use Cases 
1. **Enhanced Accuracy**: Determine the best model architecture for fish image classification 2. **Deployment Ready**: Create a user-friendly web application for real-time predictions 3. **Model Comparison**: Evaluate and compare metrics across models to select the most suitable approach 

## ✨ Features 
- **Multiple Model Architectures**: CNN from scratch + 5 pre-trained models - **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score analysis - **Data Augmentation**: Rotation, zoom, flip, and other techniques - **Interactive Web App**: Streamlit-based deployment for real-time predictions - **Model Comparison**: Detailed performance comparison with visualizations - **Confusion Matrix Analysis**: Per-class performance evaluation - **Best Model Selection**: Automatic selection and saving of the best performing model 

## 📊 Dataset 
The dataset consists of fish images categorized into folders by species. The system automatically handles: - Data loading from zip files - Train/validation splitting (80/20) - Image preprocessing and normalization - Data augmentation for training robustness 

### Data Preprocessing Steps 
1. **Image Rescaling**: Normalize pixel values to [0, 1] range 2. **Data Augmentation**: - Rotation (±30 degrees) - Width/Height shift (±20%) - Shear transformation - Zoom (±20%) - Horizontal flipping 3. **Validation Split**: 20% of data reserved for validation 

## 🤖 Models 
### 1. CNN from Scratch 
- Custom architecture with 4 convolutional blocks - BatchNormalization and Dropout for regularization - GlobalAveragePooling for dimension reduction ### 2. Transfer Learning Models - **VGG16**: Deep architecture with small filters - **ResNet50**: Skip connections for deeper networks - **MobileNet**: Lightweight architecture for efficiency - **InceptionV3**: Multi-scale feature extraction - **EfficientNetB0**: Compound scaling for optimal performance All transfer learning models use: - Pre-trained ImageNet weights - Frozen base layers - Custom classification head - Fine-tuning capabilities 

## 🚀 Installation 

### Prerequisites
bash
Python 3.8+

### Required Libraries
bash
pip install tensorflow>=2.8.0
pip install streamlit>=1.25.0
pip install scikit-learn>=1.0.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
pip install pandas>=1.3.0
pip install numpy>=1.21.0
pip install Pillow>=8.0.0
pip install joblib>=1.1.0

### Clone Repository
bash
git clone https://github.com/yourusername/fish-classification.git
cd fish-classification
pip install -r requirements.txt

## 📖 Usage ### 1. Training Models
python
from fish_classifier import FishClassifier

# Initialize classifier
classifier = FishClassifier(
    data_path="fish_dataset",
    img_height=224,
    img_width=224,
    batch_size=32
)

# Run complete training pipeline
comparison_results, best_model_name, best_model = classifier.run_complete_pipeline(
    zip_path=r"C:\Users\sathishkumar\Downloads\Fish_image_classification\Data",  # Path to your dataset zip file
    epochs=50
)
### 2. Model Evaluation
python
from model_evaluation import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator(models_dir="./", results_dir="evaluation_results")

# Define model files
model_files = {
    'VGG16': 'VGG16_best.h5',
    'ResNet50': 'ResNet50_best.h5',
    # ... other models
}

# Load and evaluate models
evaluator.load_models(model_files)
evaluator.evaluate_all_models(test_generator)
evaluator.generate_summary_report()

### 3. Streamlit Deployment
bash
streamlit run streamlit_app.py

## 📁 Project Structure
├── Data 
│ train          
│ test           
│ validation       
├── Notebook
│ fish_image_classification.ipynb         
├── app.py
├── Model evaluation and comparison.py
├── models/                    # 
│
│
│
├──requirements.txt
├──Readme.md

## 📈 Results 

### Model Performance Comparison 

| Model | Accuracy | Precision | Recall | F1-Score | Size (MB) | |-------|----------|-----------|--------|----------|-----------| | EfficientNetB0 | 94.2% | 94.1% | 94.2% | 94.1% | 20.1 | | ResNet50 | 92.8% | 92.9% | 92.8% | 92.8% | 97.8 | | VGG16 | 91.5% | 91.6% | 91.5% | 91.5% | 57.2 | | InceptionV3 | 90.3% | 90.4% | 90.3% | 90.3% | 91.7 | | MobileNet | 89.7% | 89.8% | 89.7% | 89.7% | 13.4 | | CNN Scratch | 86.2% | 86.3% | 86.2% | 86.2% | 45.6 | 

### Key Insights 
- **Best Overall Performance**: EfficientNetB0 achieves highest accuracy with reasonable model size - **Most Efficient**: MobileNet offers good performance with smallest model size - **Custom CNN**: Competitive performance considering it's built from scratch 

## 🌐 Deployment 

### Streamlit Web Application Features 
- **Image Upload**: Drag-and-drop or browse to upload fish images - **Real-time Prediction**: Instant classification with confidence scores - **Top-N Predictions**: Display multiple possible classifications - **Confidence Visualization**: Interactive charts showing prediction probabilities - **Model Information**: Display model architecture and class information 

### Deployment Steps 
1. Ensure trained models are in the correct directory 
2. Install required dependencies 
3. Run the Streamlit application:
bash
   streamlit run streamlit_app.py
4. Access the web interface at http://localhost:8501 
### Production Deployment Options 
- **Streamlit Cloud**: Direct deployment from GitHub 
- **Heroku**: Cloud platform deployment 
- **AWS/GCP**: Cloud service deployment 
- **Docker**: Containerized deployment 

## 🔧 Configuration 

### Hyperparameters
python
# Image dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Data augmentation parameters
ROTATION_RANGE = 30
ZOOM_RANGE = 0.2
SHIFT_RANGE = 0.2

### Model Selection Criteria 
- **Accuracy**: Primary metric for model selection 
- **Model Size**: Consider deployment constraints 
- **Inference Speed**: Real-time prediction requirements 
- **Training Time**: Development efficiency 

## 📊 Monitoring and Evaluation 

### Training Metrics 
- Training/Validation Accuracy 
- Training/Validation Loss 
- Learning Rate Schedule 
- Early Stopping Patience 

### Evaluation Metrics 
- **Accuracy**: Overall classification accuracy 
- **Precision**: Per-class and weighted average 
- **Recall**: Per-class and weighted average 
- **F1-Score**: Harmonic mean of precision and recall 
- **Confusion Matrix**: Detailed per-class performance 

### Visualization Features 
- Training history plots 
- Confusion matrices 
- Model comparison charts 
- Prediction probability distributions - Performance radar charts 

### Development Guidelines 
- Follow PEP 8 style guidelines 
- Add comprehensive docstrings 
- Include unit tests for new features - Update documentation for any changes 

## 💻 Tech Stacks

- 🐍 **Python** → Core programming language for model development and data preprocessing.  
- 🧠 **TensorFlow / Keras** → Building, training, and fine-tuning deep learning models (CNN, transfer learning).  
- 📊 **Pandas, NumPy** → Data manipulation, preprocessing, and numerical computations.  
- 📈 **Matplotlib, Seaborn** → Visualization of dataset distribution, training performance, and model evaluation.  
- 🌐 **Streamlit** → Deployment of the model with an interactive user interface for fish image classification.  
- 🔧 **Git / GitHub** → Version control and project collaboration.  
- 🖼️ **Pillow, Joblib** → Image preprocessing and saving/loading trained models efficiently.  

## 🙏 Acknowledgments 
- TensorFlow team for the excellent deep learning framework 
- Streamlit team for the intuitive web app framework 
- ImageNet dataset for pre-trained model weights 
- Open source community for various tools and libraries 
- Guvi mentors for supporting me throughout the project completion 

## 🏁 Conclusion

This project successfully demonstrates a multiclass fish species classification system using CNN + transfer learning.

♠ EfficientNetB0 achieved the best accuracy (94.2%).

♠ Streamlit integration makes deployment easy and user-friendly.

♠ The solution can support fisheries, marine research, and ecological monitoring. 🌊🐟

## 📞 Contact 
For questions, suggestions, or collaborations: 
- **Email**: malathisathish2228@gmail.com 
- **GitHub**: [@Malathisathish](https://github.com/malathisathish/Multiclass-fish-image-classification) 
- **LinkedIn**: [Malathisathis]("linkedin.com/in/malathi-y-datascience") --- 

**Built with ❤️ by malathi using TensorFlow and Streamlit**
