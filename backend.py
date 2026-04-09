from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import pickle
import numpy as np
from typing import Literal

app = FastAPI()
# 🔹 Input Schema (Mobile Payment Fraud)
class Transaction(BaseModel):
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    transaction_type: Literal[0,1,2,3,4]  # Encoded (PAYMENT, TRANSFER, CASH_OUT etc.)

# 🔹 Load Models
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# 🔹 Home Page (Frontend)
@app.get("/", response_class=HTMLResponse)
def serve_home():
    with open("index.html", "r") as f:
        return f.read()

# 🔹 API Root
@app.get("/api")
def api_info():
    return {"message": "Mobile Payment Fraud Detection API 🚀"}

# 🔹 Prediction Endpoint
@app.post("/predict/")
def predict(transaction: Transaction):
    
    # Convert input to array
    data = np.array([
        transaction.amount,
        transaction.oldbalanceOrg,
        transaction.newbalanceOrig,
        transaction.oldbalanceDest,
        transaction.newbalanceDest,
        transaction.transaction_type
    ]).reshape(1, -1)

    # Scale data
    data_scaled = scaler.transform(data)

    # Predict
    prediction = model.predict(data_scaled)
    probability = model.predict_proba(data_scaled)[0][1]

    return {
        "prediction": "Fraud 🚨" if prediction[0] == 1 else "Safe ✅",
        "fraud_probability": f"{probability*100:.2f}%"
    }

