# Disease Diagnosis Classification

## Overview

Disease Diagnosis Classification is a Machine Learning-based healthcare analytics project that predicts the likelihood of multiple diseases using patient medical data. The system combines data preprocessing, exploratory data analysis (EDA), model training, evaluation, and deployment through an interactive Streamlit dashboard.

The project currently supports:

* Heart Disease Prediction
* Diabetes Prediction
* Liver Disease Prediction
* Breast Cancer Analysis

## Features

* Interactive Streamlit Dashboard
* Multi-Disease Selection Interface
* Data Preprocessing and Feature Engineering
* Model Training and Evaluation
* Confusion Matrix Visualization
* ROC Curve Analysis
* Saved Models for Deployment
* Healthcare Analytics Dashboard

## Project Structure

```text
disease-diagnosis-classification/

├── app/
│   └── app.py

├── data/
│   ├── diabetes.csv
│   ├── heart.csv
│   └── Liver_Patient_Dataset.csv

├── models/
│   ├── heart_model.pkl
│   ├── heart_scaler.pkl
│   ├── diabetes_model.pkl
│   ├── liver_model.pkl
│   ├── liver_scaler.pkl
│   ├── breast_cancer_model.pkl
│   └── breast_cancer_scaler.pkl

├── notebooks/
│   ├── diabetes.ipynb
│   ├── heart_disease.ipynb
│   ├── liver_disease.ipynb
│   └── breast_cancer_analysis.ipynb

├── outputs/
│   ├── confusion matrices
│   ├── roc curves
│   └── model_results.txt

└── README.md
```

## Machine Learning Models

| Disease       | Best Model          | Accuracy |
| ------------- | ------------------- | -------- |
| Heart Disease | Logistic Regression | 85.24%   |
| Diabetes      | Random Forest       | 88.31%   |
| Liver Disease | Logistic Regression | 65.79%   |
| Breast Cancer | Logistic Regression | 97.37%   |

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Joblib
* Streamlit
* Jupyter Notebook

## Dashboard

The Streamlit dashboard allows users to:

* Select a disease for analysis
* Enter patient information
* Generate disease predictions
* View model performance metrics
* Explore healthcare analytics

## Results

* Developed machine learning models for four healthcare datasets.
* Achieved up to 97.37% accuracy on Breast Cancer classification.
* Built an interactive dashboard for disease prediction and analytics.
* Generated ROC curves and confusion matrices for model evaluation.

## Future Improvements

* Integrate AI-powered medical image analysis using MRI, CT Scan, Mammogram, and X-ray datasets.
* Enable medical scan upload functionality for automated disease prediction.
* Deploy the application on Streamlit Cloud for public access.
* Implement Explainable AI (XAI) using SHAP and LIME for prediction interpretability.
* Add disease risk probability scores and confidence levels.
* Generate automated healthcare reports with prediction summaries.
* Incorporate lifestyle factors such as smoking habits, alcohol consumption, physical activity, sleep patterns, and dietary habits.
* Develop an intelligent health assistant for personalized health recommendations.
* Build an advanced healthcare analytics dashboard with disease trend visualization.
* Expand the platform to support additional diseases such as Kidney Disease, Parkinson's Disease, Lung Cancer, and Alzheimer's Disease.

## Author

G. R. Harshitha

Indian Institute of Technology Madras
