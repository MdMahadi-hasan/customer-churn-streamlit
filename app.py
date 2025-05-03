
import streamlit as st
import joblib
import pandas as pd

# Load the trained model pipeline
model = joblib.load("churn_model_xgb_balanced.pkl")

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📉 Customer Churn Prediction Dashboard")
st.markdown("Predict whether a customer is likely to churn based on service and demographic information.")

with st.form("churn_form"):
    st.header("📋 Customer Information")

    input_dict = {
        "gender": st.selectbox("Gender", ["Male", "Female"]),
        "SeniorCitizen": st.selectbox("Senior Citizen", [0, 1]),
        "Partner": st.selectbox("Has Partner", ["Yes", "No"]),
        "Dependents": st.selectbox("Has Dependents", ["Yes", "No"]),
        "tenure": st.slider("Tenure (months)", 0, 72, 12),
        "PhoneService": st.selectbox("Phone Service", ["Yes", "No"]),
        "MultipleLines": st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"]),
        "InternetService": st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
        "OnlineSecurity": st.selectbox("Online Security", ["Yes", "No", "No internet service"]),
        "OnlineBackup": st.selectbox("Online Backup", ["Yes", "No", "No internet service"]),
        "DeviceProtection": st.selectbox("Device Protection", ["Yes", "No", "No internet service"]),
        "TechSupport": st.selectbox("Tech Support", ["Yes", "No", "No internet service"]),
        "StreamingTV": st.selectbox("Streaming TV", ["Yes", "No", "No internet service"]),
        "StreamingMovies": st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"]),
        "Contract": st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"]),
        "PaperlessBilling": st.selectbox("Paperless Billing", ["Yes", "No"]),
        "PaymentMethod": st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ]),
        "MonthlyCharges": st.number_input("Monthly Charges", min_value=0.0, value=70.0),
        "TotalCharges": st.number_input("Total Charges", min_value=0.0, value=3000.0)
    }

    submit = st.form_submit_button("Predict Churn")

if submit:
    input_df = pd.DataFrame([input_dict])

    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("🔍 Prediction Result")
        if prediction == 1:
            st.error(f"This customer is likely to churn. (Probability: {probability:.2%})")
        else:
            st.success(f"This customer is not likely to churn. (Probability: {probability:.2%})")
    except Exception as e:
        st.error("🚨 Error during prediction. Please check input format or model compatibility.")
        st.code(str(e))
