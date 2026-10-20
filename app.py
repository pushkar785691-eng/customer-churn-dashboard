"""Interactive Streamlit dashboard for individual churn-risk scoring."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("artifacts/churn_model.joblib")

st.set_page_config(page_title="Customer Churn Risk", page_icon="📉", layout="wide")
st.title("Customer Churn Risk Dashboard")
st.caption("Portfolio demonstration using the public IBM Telco Customer Churn sample.")

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python -m src.churn_pipeline` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.sidebar:
    st.header("Customer profile")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone service", ["Yes", "No"])
    multiple = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
    backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
    protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
    support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly = st.number_input("Monthly charges", 0.0, 200.0, 70.0, step=1.0)
    total = st.number_input("Total charges", 0.0, 20000.0, float(monthly * tenure), step=10.0)

record = pd.DataFrame(
    [
        {
            "gender": gender,
            "SeniorCitizen": int(senior == "Yes"),
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }
    ]
)

probability = float(model.predict_proba(record)[0, 1])
risk = "High" if probability >= 0.60 else "Medium" if probability >= 0.30 else "Low"
color = {"High": "red", "Medium": "orange", "Low": "green"}[risk]

left, right = st.columns(2)
left.metric("Predicted churn probability", f"{probability:.1%}")
right.markdown(f"### Risk band: :{color}[{risk}]")
st.progress(probability)

st.subheader("Suggested retention action")
if risk == "High":
    st.write("Prioritize outreach, investigate service issues, and offer a contract or support bundle.")
elif risk == "Medium":
    st.write("Monitor the account and present a targeted loyalty or autopay incentive.")
else:
    st.write("No immediate intervention; continue routine engagement.")

st.info(
    "This is an educational portfolio model, not a production decision system. "
    "Predictions should be reviewed by a human and monitored for drift and fairness."
)

