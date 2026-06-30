from fastapi import FastAPI
import pandas as pd
import joblib
from src.config import MODEL_PATH

from api.schema import CustomerData, PredictionResponse

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0"
)

model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return{
        "message": "Churn prediction API running"
    }



@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    customer: CustomerData
):
    customer_df = pd.DataFrame(
        [customer.model_dump()]
    )

    probability = model.predict_proba(customer_df)[0][1]

    if probability >= 0.70:
        risk = "High Risk"

    elif probability >= 0.40:
        risk = "Medium Risk"

    else:
        risk = "Low Risk"

    if probability >= 0.5:
        prediction = "Likely To Churn"

    else:
        prediction = "Not Likely To Churn"

    return{
        "prediction": prediction,
        "churn_probability":
        round(float(probability),4),
        "risk_tier":risk
    }

@app.get("/model-info")
def model_info():

    return {
        "model_name": "Logistic Regression",
        "target": "Customer Churn",
        "risk_thresholds": {
            "High Risk": ">= 0.70",
            "Medium Risk": "0.40 - 0.69",
            "Low Risk": "< 0.40"
        }
    }