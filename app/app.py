import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Multi-Disease Diagnosis System",
    page_icon="🏥"
)

st.title("🏥 Multi-Disease Diagnosis System")

st.write("Machine Learning based healthcare diagnosis platform")

disease = st.selectbox(
    "Select Disease",
    [
        "Diabetes",
        "Heart Disease",
        "Liver Disease",
        "Breast Cancer"
    ]
)

# ==========================
# DIABETES
# ==========================

if disease == "Diabetes":

    model = joblib.load("models/diabetes_model.pkl")

    st.header("🩸 Diabetes Prediction")

    pregnancies = st.number_input("Pregnancies", min_value=0, value=0)
    glucose = st.number_input("Glucose", min_value=0.0, value=100.0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, value=80.0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, value=20.0)
    insulin = st.number_input("Insulin", min_value=0.0, value=80.0)
    bmi = st.number_input("BMI", min_value=0.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
    age = st.number_input("Age", min_value=1, value=30)

    if st.button("Predict Diabetes"):

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
    st.write("Best Model: Random Forest")
    st.write("Accuracy: 88.31%")

# ==========================
# HEART DISEASE
# ==========================

elif disease == "Heart Disease":

    st.header("❤️ Heart Disease Prediction")

    heart_model = joblib.load("models/heart_model.pkl")
    heart_scaler = joblib.load("models/heart_scaler.pkl")

    age = st.number_input("Age", min_value=1, value=45)
    sex = st.selectbox("Sex", ["Female", "Male"])
    cp_option = st.selectbox(
    "Chest Pain Type",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-Anginal Pain",
        "Asymptomatic"
    ]
    )

    cp_mapping = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-Anginal Pain": 2,
    "Asymptomatic": 3
    }

    cp = cp_mapping[cp_option]
    trestbps = st.number_input("Resting Blood Pressure", min_value=0, value=120)
    chol = st.number_input("Cholesterol", min_value=0, value=200)
    fbs = st.number_input("Fasting Blood Sugar (0/1)", min_value=0, max_value=1, value=0)
    restecg = st.number_input("Rest ECG", min_value=0, value=0)
    thalach = st.number_input("Maximum Heart Rate", min_value=0, value=150)
    exang = st.number_input("Exercise Induced Angina (0/1)", min_value=0, max_value=1, value=0)
    oldpeak = st.number_input("Old Peak", value=1.0)
    slope = st.number_input("Slope", min_value=0, value=1)
    ca = st.number_input("CA", min_value=0, value=0)
    thal = st.number_input("Thal", min_value=0, value=2)

    sex = 1 if sex == "Male" else 0

    if st.button("Predict Heart Disease"):

        input_data = pd.DataFrame([[
            age, sex, cp, trestbps, chol,
            fbs, restecg, thalach,
            exang, oldpeak, slope,
            ca, thal
        ]], columns=[
            'age','sex','cp','trestbps','chol',
            'fbs','restecg','thalach',
            'exang','oldpeak','slope',
            'ca','thal'
        ])

        input_scaled = heart_scaler.transform(input_data)

        prediction = heart_model.predict(input_scaled)[0]

        if prediction == 1:
            st.error("⚠️ Heart Disease Risk Detected")
        else:
            st.success("✅ No Heart Disease Risk Detected")

    st.markdown("---")
    st.write("Best Model: Logistic Regression")
    st.write("Accuracy: 85.24%")

# ==========================
# LIVER DISEASE
# ==========================

elif disease == "Liver Disease":

    st.header("🧪 Liver Disease Prediction")

    liver_model = joblib.load("models/liver_model.pkl")
    liver_scaler = joblib.load("models/liver_scaler.pkl")

    age = st.number_input("Age", min_value=1, value=45)

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    tot_bilirubin = st.number_input("Total Bilirubin", value=1.0)
    direct_bilirubin = st.number_input("Direct Bilirubin", value=0.3)
    tot_proteins = st.number_input("Total Proteins", value=200.0)
    albumin = st.number_input("Albumin", value=35.0)
    ag_ratio = st.number_input("A/G Ratio", value=1.0)
    sgpt = st.number_input("SGPT", value=30.0)
    sgot = st.number_input("SGOT", value=30.0)
    alkphos = st.number_input("Alkaline Phosphatase", value=200.0)

    gender = 1 if gender == "Male" else 0

    if st.button("Predict Liver Disease"):

        input_data = pd.DataFrame([[
            age,
            gender,
            tot_bilirubin,
            direct_bilirubin,
            tot_proteins,
            albumin,
            ag_ratio,
            sgpt,
            sgot,
            alkphos
        ]], columns=[
            'age',
            'gender',
            'tot_bilirubin',
            'direct_bilirubin',
            'tot_proteins',
            'albumin',
            'ag_ratio',
            'sgpt',
            'sgot',
            'alkphos'
        ])

        input_scaled = liver_scaler.transform(input_data)

        prediction = liver_model.predict(input_scaled)[0]

        if prediction == 1:
            st.error("⚠️ Liver Disease Risk Detected")
        else:
            st.success("✅ Healthy Liver")

    st.markdown("---")
    st.write("Best Model: Logistic Regression")
    st.write("Accuracy: 65.79%")
# ==========================
# RESULTS SUMMARY
# ==========================

st.markdown("---")

st.subheader("📊 Model Performance Summary")

results = pd.DataFrame(
    {
        "Disease": [
            "Heart Disease",
            "Diabetes",
            "Liver Disease",
            "Breast Cancer"
        ],
        "Best Model": [
            "Logistic Regression",
            "Random Forest",
            "Logistic Regression",
            "Logistic Regression"
        ],
        "Accuracy (%)": [
            85.24,
            88.31,
            65.79,
            97.37
        ]
    }
)

st.dataframe(results)