# 🏦 Customer Churn Predictor

![Churn Prediction](https://img.shields.io/badge/AI-Customer%20Churn-purple)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)

## Overview
ChurnMonitor is an AI-powered application that predicts whether bank customers are likely to leave based on their profile data. The model analyzes customer characteristics and behaviors to identify potential churn risk, enabling proactive customer retention strategies.

## Usage
1. Input customer details using the provided form controls
2. Click the "Predict" button to get the churn probability
3. View the prediction result with visual indicators

##  Live Demo
Experience the prediction model in action: [ChurnMonitor App](https://churnmonitoring.streamlit.app/)

##  Tech Stack

### ML Framework
- **TensorFlow**: Built and trained an Artificial Neural Network (ANN) to predict customer churn
- **Keras**: Used for constructing the deep learning model architecture
- **Scikit-learn**: Employed for data preprocessing, encoding, scaling, and evaluation

### Data Processing
- **NumPy**: Utilized for numerical operations and array manipulations
- **Pandas**: Used for data manipulation and analysis
- **Pickle**: Implemented for model serialization and persistence

### Web Application
- **Streamlit**: Developed the interactive web interface for model deployment
- **Streamlit-Lottie**: Integrated engaging animations to enhance user experience

### Visualization
- **Matplotlib**: Created data visualizations for model performance analysis

### Model Optimization
- **TensorBoard**: Monitored and visualized training metrics
- **Keras-Tuner**: Performed hyperparameter optimization

##  Features

### Technical Features
- **Deep Learning Model**: Multi-layer Artificial Neural Network trained to recognize patterns associated with customer churn
- **Data Preprocessing Pipeline**: Automated encoding of categorical features and scaling of numerical values
- **Model Persistence**: Saved trained model and preprocessing components for deployment
- **Input Validation**: Ensures data quality and consistency for reliable predictions

### User Interface Features
- **Interactive Input Form**: Easy-to-use sliders and selectors for entering customer data
- **Real-time Prediction**: Instant churn probability calculation
- **Visual Feedback**: Dynamic animations and color-coded results based on prediction outcome
- **Responsive Design**: Clean interface that works across different devices
- **Custom Styling**: Professional appearance with custom fonts and gradient buttons

##  Model Architecture
The churn prediction model is built using a TensorFlow-based Artificial Neural Network with:
- Input layer matching the feature dimensions
- Hidden layers with ReLU activation functions
- Dropout layers to prevent overfitting
- Binary classification output with sigmoid activation
- Adam optimizer and binary crossentropy loss function

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/churn-monitor.git
cd churn-monitor

# Create and activate virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run App.py
```


