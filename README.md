# Disease Diagnosis Classification

## Overview

Disease Diagnosis Classification is a Machine Learning project designed to predict disease outcomes using patient medical data. The system analyzes multiple health-related attributes and classifies whether a patient is likely to have a disease.

The project compares several machine learning algorithms and evaluates their performance using industry-standard metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

---

## Objectives

- Predict disease presence using patient medical records.
- Compare multiple machine learning algorithms.
- Identify the most effective model for disease classification.
- Improve healthcare decision-making through data-driven insights.

---

## Dataset

The dataset contains 500+ patient records and 30+ medical attributes, including:

- Age
- Blood Pressure
- Glucose Level
- BMI
- Insulin
- Cholesterol
- Heart Rate
- Other clinical measurements

Target Variable:

- 0 = No Disease
- 1 = Disease Present

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## Project Workflow

### 1. Data Preprocessing

- Missing value handling
- Duplicate removal
- Feature scaling
- Data normalization

### 2. Exploratory Data Analysis

- Distribution analysis
- Correlation heatmaps
- Feature relationships
- Disease prevalence analysis

### 3. Model Development

The following algorithms were trained and evaluated:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree

### 4. Model Evaluation

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

---

## Results

| Model | Accuracy |
|---------|---------|
| Logistic Regression | 95%+ |
| Random Forest | 94%+ |
| SVM | 93%+ |
| KNN | 91%+ |
| Decision Tree | 89%+ |

Best Model: Logistic Regression

Performance Achieved:

- Accuracy: 95%+
- ROC-AUC: 0.95+
- High Precision and Recall
- Strong Generalization Performance

---

## Key Features

- Automated disease prediction
- Multi-model comparison
- Healthcare data analysis
- Feature importance evaluation
- Performance visualization

---

## Future Improvements

- Deep Learning implementation
- Web application deployment using Streamlit
- Real-time patient risk prediction
- Integration with healthcare management systems

---

## Repository Structure

```
disease-diagnosis-classification/
│
├── data/
│   └── disease_dataset.csv
│
├── notebooks/
│   └── disease_diagnosis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── models/
│   └── best_model.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Author

**G.R. Harshitha**

B.Tech Mechanical Engineering  
Indian Institute of Technology Madras

GitHub: https://github.com/harshitha121124
