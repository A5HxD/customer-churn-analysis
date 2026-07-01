# Customer Churn Prediction & MLOps Pipeline

An end-to-end Machine Learning project that predicts telecom customer churn using customer demographics, service usage, and billing information. The project covers the complete ML lifecycle—from data analysis and model development to experiment tracking, API deployment, and containerization using modern MLOps practices.

---

## Project Overview

Customer churn is one of the biggest challenges faced by subscription-based businesses. This project predicts whether a customer is likely to churn using historical telecom customer data and segments customers into different risk tiers to support targeted retention strategies.

The project goes beyond model development by implementing a production-ready inference pipeline using FastAPI, MLflow, and Docker.

---

## Problem Statement

Build an end-to-end customer churn prediction pipeline that analyzes telecom customer behavior, predicts churn probability, and deploys the trained model as a scalable REST API using modern MLOps tools.

---

## Dataset

**Dataset:** Telco Customer Churn Dataset

- Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Records: 7,043 customers
- Features: 21 customer attributes
- Target Variable: `Churn` (Yes / No)

The dataset contains:

- Customer demographics
- Account information
- Services subscribed
- Billing information
- Customer tenure
- Churn status

---

# Tech Stack

### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn
- Plotly

### Machine Learning

- Scikit-Learn
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

### MLOps

- MLflow
- FastAPI
- Docker

---

# Project Structure

```text
customer-churn-analysis/
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── schema.py
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   └── logistic_regression_pipeline.pkl
│
├── notebooks/
│   └── churn_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── mlruns/
│
├── Dockerfile
├── requirements.txt
├── .dockerignore
└── README.md
```

---

# Exploratory Data Analysis

Performed detailed exploratory data analysis including:

- Dataset overview
- Missing value analysis
- Class imbalance analysis
- Summary statistics
- Correlation analysis
- Feature relationship visualization

---

# Data Preprocessing

Implemented a reusable preprocessing pipeline including:

- Removed irrelevant features (`customerID`)
- Converted `TotalCharges` to numeric
- Missing value handling
- One-Hot Encoding using `OneHotEncoder`
- Feature scaling using `StandardScaler`
- Stratified Train-Test Split (80:20)

The preprocessing pipeline is integrated using Scikit-Learn's `ColumnTransformer` and `Pipeline`.

---

# Model Development

Three classification models were trained and evaluated.

| Model | Purpose |
|--------|----------|
| Logistic Regression | Baseline Linear Model |
| Random Forest | Ensemble Tree Model |
| Gradient Boosting | Boosted Ensemble |

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Hyperparameter tuning was performed using `RandomizedSearchCV`.

---

# 🏆 Best Performing Model

**Logistic Regression**

Selected because it achieved the best balance of:

- Accuracy
- Precision
- Recall
- ROC-AUC

while maintaining excellent interpretability and faster inference.

---

# 📈 Customer Risk Segmentation

Customers are segmented using predicted churn probability.

| Risk Tier | Probability |
|-----------|-------------|
| 🔴 High Risk | ≥ 0.70 |
| 🟡 Medium Risk | 0.40 – 0.69 |
| 🟢 Low Risk | < 0.40 |

Each tier is analyzed based on:

- Monthly Charges
- Customer Tenure
- Contract Type

---

# 📊 Visualizations

The project includes multiple visualizations including:

- Feature Importance
- Churn by Contract Type
- Tenure Distribution
- ROC Curve Comparison
- Confusion Matrices
- Risk Tier Distribution
- Customer Risk Analysis

---

# 🔄 Machine Learning Pipeline

The project has been modularized into reusable components.

```
Load Data
      │
      ▼
Data Cleaning
      │
      ▼
Preprocessing Pipeline
      │
      ▼
Train/Test Split
      │
      ▼
Model Training
      │
      ▼
Evaluation
      │
      ▼
Save Model
```

---

# 📋 MLflow Integration

MLflow is used to track machine learning experiments.

Tracked information includes:

- Model Parameters
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Saved Model Artifacts

Launch MLflow UI:

```bash
mlflow ui
```

Open:

```
http://127.0.0.1:5000
```

---

# 🚀 FastAPI Inference Service

The trained model is deployed locally using FastAPI.

Run:

```bash
uvicorn api.main:app --reload
```

Swagger Documentation:

```
http://127.0.0.1:8000/docs
```

Available endpoints:

```
GET /

POST /predict
```

The API returns:

- Churn Prediction
- Churn Probability
- Risk Tier

---

# 🐳 Docker Support

The application has been containerized using Docker.

Build Image

```bash
docker build -t churn-api .
```

Run Container

```bash
docker run -p 8000:8000 churn-api
```

---

# 📊 Current Architecture

```
                User
                  │
                  ▼
            FastAPI API
                  │
                  ▼
     Logistic Regression Pipeline
                  │
                  ▼
      Scikit-Learn Preprocessor
                  │
                  ▼
        Customer Churn Prediction
```

---

# 🔮 Future Improvements

- Deploy on AWS EC2
- Add Docker Compose
- Implement CI/CD using GitHub Actions
- Model Monitoring
- MLflow Model Registry
- Authentication for API
- Interactive Streamlit Dashboard
- Continuous Model Retraining

---

# 💻 Installation

Clone the repository

```bash
git clone https://github.com/A5HxD/customer-churn-analysis.git

cd customer-churn-analysis
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run training

```bash
python src/train.py
```

Launch API

```bash
uvicorn api.main:app --reload
```

---

# 📌 Key Highlights

- End-to-End Machine Learning Pipeline
- Modular Project Structure
- Production-Ready Preprocessing Pipeline
- Experiment Tracking with MLflow
- REST API using FastAPI
- Containerized using Docker
- Customer Risk Segmentation
- Business-Oriented Insights
- Clean, Scalable Architecture

---

# 👨‍💻 Author

**Ayush Prasad**

GitHub: https://github.com/A5HxD

LinkedIn: https://www.linkedin.com/in/ayushprasad

---
