import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("churn_model_xgb_balanced.pkl")

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📉 Customer Churn Prediction Dashboard")
st.markdown("Predict whether a customer is likely to churn based on service and demographic information.")

# Input form for customer features
with st.form("churn_form"):
    st.header("📋 Customer Information")

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protect = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total = st.number_input("Total Charges", min_value=0.0, value=3000.0)

    submit = st.form_submit_button("Predict Churn")

if submit:
    # Create input DataFrame
    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protect,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    # Predict churn
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error(f"This customer is likely to churn. (Probability: {probability:.2%})")
    else:
        st.success(f"This customer is not likely to churn. (Probability: {probability:.2%})")