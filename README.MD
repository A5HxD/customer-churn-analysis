# Customer Churn Prediction & Risk Segmentation

An end-to-end Machine Learning project focused on predicting telecom customer churn, identifying key churn drivers, segmenting customers into risk tiers, and generating business-oriented retention insights using the Telco Customer Churn dataset.

---

## Business Problem

Customer churn is one of the most critical challenges faced by subscription-based businesses such as telecom companies. Acquiring new customers is significantly more expensive than retaining existing ones, making churn prediction an important business problem.

The objective of this project is to:
- predict whether a customer is likely to churn,
- identify the major factors influencing churn,
- segment customers into risk groups,
- and provide actionable business recommendations for improving customer retention.

---

## Dataset

Dataset Used:
[Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Dataset contains:
- Customer demographics
- Subscription information
- Billing details
- Internet and support services
- Churn status

Total Records: **7043 customers**

---

# Project Workflow

## 1. Data Loading & Exploratory Data Analysis (EDA)

Performed:
- Dataset inspection
- Missing value analysis
- Statistical summary
- Correlation analysis
- Churn imbalance analysis

Key Findings:
- Customers with lower tenure churned more frequently
- Month-to-month contracts had significantly higher churn rates
- Support-related services reduced churn probability

---

## 2. Data Preprocessing & Feature Engineering

### Preprocessing
- Converted `TotalCharges` to numeric format
- Handled coercion errors and missing values
- Applied One-Hot Encoding to categorical variables
- Scaled numerical features using `StandardScaler`

### Feature Engineering
Created additional features such as:
- `ChargesPerMonth`
- `SeniorWithNoSupport`

### Train-Test Split
- 80% Training
- 20% Testing
- Stratified split used to preserve churn ratio

---

## 3. Model Training & Evaluation

The following classification models were trained and compared:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

### Evaluation Metrics
Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### Best Model
✅ **Logistic Regression** achieved the best overall performance.

Reason:
- Better balance between Recall and F1-Score
- Strong interpretability
- Stable probability outputs
- Suitable for structured telecom data after One-Hot Encoding

### Hyperparameter Tuning
- Performed using `RandomizedSearchCV`

---

## 4. Customer Risk Segmentation

Customers were segmented using churn probability scores:

| Risk Tier | Probability Range |
|---|---|
| High Risk | ≥ 0.70 |
| Medium Risk | 0.40 – 0.69 |
| Low Risk | < 0.40 |

### High-Risk Customer Patterns
High-risk customers commonly had:
- Month-to-month contracts
- Higher monthly charges
- Lower tenure
- Fewer support services

---

# Visualizations

The project includes multiple business-oriented visualizations:

- Feature Importance Chart
- Churn Rate by Contract Type
- Tenure Distribution by Churn Status
- Risk Tier Distribution Donut Chart
- ROC Curve Comparison
- Confusion Matrix Comparison
- Interactive Plotly Scatter Plot

---

# Key Business Insights

### 1. Contract Type is the Strongest Churn Driver
Customers on month-to-month contracts showed the highest churn probability.

### 2. Customer Tenure Strongly Impacts Retention
Customers with shorter tenure were significantly more likely to churn.

### 3. Support Services Improve Retention
Customers using:
- TechSupport
- OnlineSecurity
- DeviceProtection

were less likely to churn.

---

# Business Recommendations

### Recommendation 1
Improve early-stage customer onboarding and retention programs to reduce churn during the first few months.

### Recommendation 2
Encourage customers to shift toward long-term contracts using discounts, loyalty rewards, and bundled service offerings.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Jupyter Notebook

---

# Repository Structure

```text
customer-churn-prediction/
│
├── data/
│   └── telco_customer_churn.csv
│
├── models/
│
├── notebooks/
│   └── churn_analysis.ipynb
│
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── README.md
|
└── requirements.txt
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/A5HxD/customer-churn-analysis.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook
```

---

# Future Improvements

Possible future enhancements:
- XGBoost or LightGBM implementation
- SMOTE for class imbalance handling
- Streamlit deployment
- Real-time churn prediction dashboard
- Time-series customer behavior analysis

---

# Conclusion

This project demonstrates how machine learning can be applied to solve real-world business problems such as telecom customer churn. By combining predictive modeling, customer segmentation, and business analytics, the project provides actionable insights that can help organizations improve customer retention and reduce revenue loss.
