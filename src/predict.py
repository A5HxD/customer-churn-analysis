## Used for local testing

import joblib

import pandas as pd

from config import MODEL_PATH

model = joblib.load(
    MODEL_PATH
)


sample_customer ={
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 42.5,
    "TotalCharges": 489.76
}

customer_df = pd.DataFrame([sample_customer])

probability = model.predict_proba(customer_df)[0][1]

if probability >= 0.70:
    risk = "High risk"

elif probability >= 0.40:
    risk = "Medium risk"

else:
    risk = "Low risk"

print(f"Risk Tier: {risk}")