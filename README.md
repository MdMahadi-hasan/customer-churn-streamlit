# Create a README.md file content
readme_content = """
# Customer Churn Prediction Dashboard

This project is a deployable Streamlit dashboard that predicts customer churn based on demographic and service usage features. The model is trained on the Telco Customer Churn dataset using XGBoost with class imbalance handled via manual oversampling.

## 📁 Files Included

- `churn_model_training.py` – Full data preprocessing, oversampling, model training and saving pipeline.
- `churn_model_xgb_balanced.pkl` – Trained model saved as a scikit-learn pipeline for use in deployment.
- `app.py` – Streamlit dashboard to interactively predict churn.
- `README.md` – This file.

## 🔍 Dataset

Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## 🛠 Features

- Preprocessing for numeric and categorical columns
- Manual oversampling to handle class imbalance
- Trained with XGBoost Classifier
- Streamlit app to input customer data and get churn prediction
- Shows prediction label and probability

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
