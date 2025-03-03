import streamlit as st
import pandas as pd 
import joblib
import os 
import numpy as np
from tensorflow.keras.models import load_model
import xgboost as xgb

# Load models
nn_model = load_model("nn_model.h5")
xgb_model = xgb.Booster()
xgb_model.load_model("xgb_model.json")

st.title("Customer Churn Prediction App 🚀")
st.write("This app predicts customer churn using a Neural Network + XGBoost ensemble.")

# Personal Information Section
with st.expander("📌 Personal Information", expanded=True):
    personal_inputs = {
        "gender": st.selectbox("Gender", ["Male", "Female"]),
        "SeniorCitizen": st.selectbox("Senior Citizen", [0, 1]),
        "Partner": st.selectbox("Partner", ["Yes", "No"]),
        "Dependents": st.selectbox("Dependents", ["Yes", "No"]),
        "tenure": st.number_input("Tenure"),
    }

# Technical Details Section
with st.expander("🔧 Technical Details", expanded=False):
    technical_inputs = {
        "PhoneService": st.selectbox("PhoneService", ["No", "Yes"]),
        "MultipleLines": st.selectbox("MultipleLines", ["No phone service", "No", "Yes"]),
        "InternetService": st.selectbox("InternetService", ["DSL", "Fiber optic", "No"]),
        "OnlineSecurity": st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"]),
        "OnlineBackup": st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"]),
        "DeviceProtection": st.selectbox("DeviceProtection", ["Yes", "No", "No internet service"]),
        "TechSupport": st.selectbox("TechSupport", ["Yes", "No", "No internet service"]),
        "StreamingTV": st.selectbox("StreamingTV", ["Yes", "No", "No internet service"]),
        "StreamingMovies": st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"]),
        "Contract": st.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
    }

# Financial Information Section
with st.expander("💰 Financial Information", expanded=False):
    financial_inputs = {
        "PaperlessBilling": st.selectbox("PaperlessBilling", ["Yes", "No"]),
        "PaymentMethod": st.selectbox(
            "PaymentMethod",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        ),
        "MonthlyCharges": st.number_input("MonthlyCharges"),
        "TotalCharges": st.number_input("TotalCharges"),
    }


# Load encoders
encoder_info = joblib.load("encoder.pkl")
encoders = encoder_info["encoders"]  
categorical_cols = encoder_info["features"]  

# Load scaler
scaler_info = joblib.load("scaler.pkl")
scaler = scaler_info["scaler"]  
numerical_cols = scaler_info["features"]  

if st.button("Submit"):
    # Combine inputs
    user_inputs = {**personal_inputs, **technical_inputs, **financial_inputs}
    input_df = pd.DataFrame([user_inputs])

    # Encode categorical features
    for col in categorical_cols:
        if col in input_df.columns:
            input_df[col] = encoders[col].transform(input_df[col])

    # Scale numerical features
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    st.write("Encoded and Scaled Input:")
    st.write(input_df)

    # Convert to NumPy for Neural Network
    input_np = input_df.to_numpy()

    # Neural Network Prediction
    nn_pred = nn_model.predict(input_np)[0][0]

    # XGBoost requires DMatrix for prediction
    dtest = xgb.DMatrix(input_df)
    xgb_pred = xgb_model.predict(dtest)[0]  

    # Ensemble Prediction (Simple Average)
    final_pred = (nn_pred + xgb_pred) / 2

    # Display the result with larger font
    st.subheader("Prediction Result")

    # Show probability in large font
    st.markdown(f"<h2 style='text-align: center; color: #4CAF50;'>Churn Probability: {final_pred:.2f}</h2>", unsafe_allow_html=True)

    # Show churn decision with bigger size and different colors
    if final_pred > 0.5:
        st.markdown("<h2 style='text-align: center; color: red;'>🚨 This customer is likely to churn ❌</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='text-align: center; color: green;'>✅ This customer is likely to stay 🎉</h2>", unsafe_allow_html=True)

