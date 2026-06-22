import streamlit as st
import joblib
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Multi-Disease Diagnosis System",
    page_icon="🏥"
)

# Load Diabetes Model
model = joblib.load("models/diabetes_model.pkl")

# Title
st.title("🏥 Multi-Disease Diagnosis System")

st.write(
    "Machine Learning based healthcare diagnosis platform"
)

st.header("🩸 Diabetes Prediction")

# Input Fields
pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    value=0
)

glucose = st.number_input(
    "Glucose",
    min_value=0.0,
    value=100.0
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0.0,
    value=80.0
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0.0,
    value=20.0
)

insulin = st.number_input(
    "Insulin",
    min_value=0.0,
    value=80.0
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    value=25.0
)

dpf = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    value=0.5
)

age = st.number_input(
    "Age",
    min_value=1,
    value=30
)

# Prediction
if st.button("Predict"):

    input_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    )

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ Diabetes Risk Detected")
    else:
        st.success("✅ No Diabetes Risk Detected")

st.markdown("---")

st.subheader("Model Information")

st.write("Best Model: Random Forest")

st.write("Accuracy: 88.31%")