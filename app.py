import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model + scaler + feature schema
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

st.set_page_config(page_title="Healthcare Risk Predictor", layout="centered")

st.title("Healthcare Risk Prediction System")
st.markdown("Predict patient risk using a trained Machine Learning model")

st.divider()

# --- INPUT UI ---
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 100, 50)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.slider("Cholesterol", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])

with col2:
    restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
    thalach = st.slider("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope (0–2)", [0, 1, 2])
    ca = st.selectbox("Major Vessels (0–4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal (0–3)", [0, 1, 2, 3])

st.divider()

# --- Convert Input to DataFrame ---
input_dict = {
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}

input_df = pd.DataFrame([input_dict])

# --- Apply SAME preprocessing as training ---
input_encoded = pd.get_dummies(input_df)

# Align with training columns (CRITICAL FIX)
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

# Scale
input_scaled = scaler.transform(input_encoded)

# --- Prediction ---
if st.button("Predict Risk", use_container_width=True):

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"High Risk ⚠️ | Probability: {probability:.2f}")
    else:
        st.success(f"Low Risk ✅ | Probability: {probability:.2f}")

    st.progress(float(probability))

    # --- Feature Importance ---
    if hasattr(model, "feature_importances_"):
        st.subheader("Top Risk Factors")

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:5]

        top_features = [feature_columns[i] for i in indices]
        top_importances = importances[indices]

        fig, ax = plt.subplots()
        ax.barh(top_features[::-1], top_importances[::-1])
        ax.set_title("Top Contributing Features")

        st.pyplot(fig)