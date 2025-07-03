# app.py

import streamlit as st
import pandas as pd
import joblib

# === Load Model ===
model = joblib.load("heart_model2.pkl")
model_columns = joblib.load("model_columns.pkl")

# === App UI ===
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("Heart Disease Risk Predictor")
st.markdown("Enter patient information below:")

# === Input Form ===
gender = st.selectbox("Sex", ['Male', 'Female'])
age_category = st.selectbox("Age Category", [
    '18-24', '25-29', '30-34', '35-39', '40-44', '45-49',
    '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'
])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)
smoking = st.selectbox("Do you smoke?", ['Yes', 'No'])
alcohol = st.selectbox("Do you drink alcohol?", ['Yes', 'No'])
stroke = st.selectbox("Have you had a stroke?", ['Yes', 'No'])
diff_walking = st.selectbox("Difficulty Walking?", ['Yes', 'No'])
diabetic = st.selectbox("Diabetic?", ['Yes', 'No', 'No, borderline diabetes', 'Yes (during pregnancy)'])
physical_health = st.slider("Poor Physical Health (days)", 0, 30, 0)
mental_health = st.slider("Poor Mental Health (days)", 0, 30, 0)
sleep_time = st.slider("Sleep Time (hours/night)", 0, 24, 7)
phys_act = st.selectbox("Physically Active?", ['Yes', 'No'])
gen_health = st.selectbox("General Health", ['Excellent', 'Very good', 'Good', 'Fair', 'Poor'])
kidney_disease = st.selectbox("Kidney Disease?", ['Yes', 'No'])

# === Predict Button ===
if st.button("Predict Heart Disease Risk"):
    input_data = {
        'BMI': bmi,
        'PhysicalHealth': physical_health,
        'MentalHealth': mental_health,
        'SleepTime': sleep_time,
        'Smoking': smoking,
        'AlcoholDrinking': alcohol,
        'Stroke': stroke,
        'DiffWalking': diff_walking,
        'Sex': gender,
        'AgeCategory': age_category,
        'Diabetic': diabetic,
        'PhysicalActivity': phys_act,
        'GenHealth': gen_health,
        'KidneyDisease': kidney_disease
    }

    df_input = pd.DataFrame([input_data])
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=model_columns, fill_value=0)

    pred = model.predict(df_input)[0]
    conf = model.predict_proba(df_input)[0][1]

    st.subheader("🧾 Result:")
    if pred == 1:
        st.error(f"Prediction: **Heart Disease** \n")
        st.error(f"Confidence: **{conf:.2%}**")
    else:
        st.success(f"Prediction: **No Heart Disease**  \n")
        st.success(f"Confidence: **{(1 - conf):.2%}**")
