# Healthcare Patient Risk Stratification System

## Overview

This project presents an end-to-end machine learning system for predicting patient risk levels using structured healthcare data. It covers the full lifecycle including data preprocessing, model training, evaluation, and deployment via an interactive web application.

## Objective

To classify patients into high-risk and low-risk categories based on clinical attributes, enabling data-driven healthcare decision support.

## Key Features

* End-to-end machine learning pipeline
* Model comparison and benchmarking
* Consistent preprocessing between training and inference
* Real-time predictions through an interactive web interface
* Feature importance analysis for interpretability

## Dataset

The model is trained on a structured healthcare dataset (Heart Disease dataset), containing features such as age, cholesterol levels, blood pressure, and other clinical indicators.

## Models Used

* Logistic Regression (baseline model)
* Random Forest Classifier (final model)

## Results

* ROC-AUC: ~0.94
* Accuracy improvement: ~4.5% over baseline
* Random Forest outperformed Logistic Regression by capturing non-linear feature relationships

## System Architecture

User Input → Preprocessing → Model → Prediction → Output

## Deployment

The system is deployed as an interactive web application using Streamlit. Users can input patient data and receive real-time risk predictions.

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib
* Streamlit

## Project Structure

healthcare-risk-model/
├── src/                    # Core ML pipeline code
├── models/                 # Saved model, scaler, feature schema
├── outputs/                # Metrics and plots
├── app.py                  # Streamlit application
├── main.py                 # Training pipeline
├── requirements.txt

## How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Train the model

python main.py

### 3. Run the application

python -m streamlit run app.py

## Key Learnings

* Handling feature mismatch between training and inference
* Building consistent ML pipelines using persisted artifacts
* Model evaluation using ROC-AUC and classification metrics
* Deploying machine learning models into interactive applications

## Future Improvements

* Model calibration and cross-validation
* Advanced explainability using SHAP
* Integration with real-world healthcare datasets

## Author

Jash Shah
