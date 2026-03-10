import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib

# -----------------------------
# Load saved artifacts
# -----------------------------
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="Explainable AI for Breast Cancer Diagnosis",
    layout="wide"
)

st.title("Explainable AI for Breast Cancer Diagnosis")
st.write(
    """
    This application predicts breast cancer using a trained machine learning model
    and explains the prediction using SHAP (Explainable AI).
    """
)

# -----------------------------
# Sidebar: User input
# -----------------------------
st.sidebar.header("Input Clinical Features")

input_data = {}
for feature in feature_names:
    input_data[feature] = st.sidebar.number_input(
        label=feature,
        min_value=0.0,
        value=0.0
    )

input_df = pd.DataFrame([input_data])

# -----------------------------
# Preprocess input
# -----------------------------
input_scaled = scaler.transform(input_df)
input_scaled = pd.DataFrame(input_scaled, columns=feature_names)

# -----------------------------
# Prediction
# -----------------------------
prediction = rf_model.predict(input_scaled)[0]
probability = rf_model.predict_proba(input_scaled)[0][1]

st.subheader("Prediction Result")

if prediction == 1:
    st.error(f"Malignant (Cancer Detected)\n\nProbability: {probability:.2f}")
else:
    st.success(f"Benign (No Cancer Detected)\n\nProbability: {1 - probability:.2f}")

# -----------------------------
# SHAP Explainability
# -----------------------------
st.subheader("Model Explanation (SHAP)")

explainer = shap.Explainer(rf_model, input_scaled)
shap_values = explainer(input_scaled)

# Extract SHAP values for malignant class (class = 1)
sv = shap_values.values[0, :, 1]
base_value = shap_values.base_values[0, 1]

explanation = shap.Explanation(
    values=sv,
    base_values=base_value,
    data=input_scaled.iloc[0],
    feature_names=feature_names
)

# Plot SHAP waterfall
import matplotlib.pyplot as plt

st.subheader("Model Explanation (SHAP)")

try:
    plt.clf()
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(plt.gcf())
except Exception as e:
    st.error("Unable to render SHAP explanation.")
    st.exception(e)
