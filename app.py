import os
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="CardioCare - Clinical Decision Support",
    page_icon="🫀",
    layout="wide"
)

# Cache loaded pipeline
@st.cache_resource
def load_pipeline():
    with open('heart_disease_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    pipeline = load_pipeline()
except FileNotFoundError:
    st.error("Model artifact 'heart_disease_model.pkl' not found! Please run 'python train_and_benchmark.py' first.")
    st.stop()

# Header
st.title("🫀 CardioCare: Heart Disease Risk Assessment")
st.markdown("Clinical Decision Support interface driven by a Scikit-Learn diagnostic pipeline.")
st.divider()

# Input Form Structured by Clinical Workflow
with st.form("clinical_assessment_form"):
    st.subheader("1. Patient Demographics & Baseline Vitals")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age (Years):", min_value=20, max_value=90, value=54)
        sex = st.selectbox("Biological Sex:", options=[1.0, 0.0], format_func=lambda x: "Male" if x == 1.0 else "Female")
    
    with col2:
        trestbps = st.number_input("Resting Blood Pressure (mm Hg):", min_value=80, max_value=220, value=130)
        chol = st.number_input("Serum Cholesterol (mg/dl):", min_value=100, max_value=600, value=240)
        
    with col3:
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl:",
            options=[0.0, 1.0],
            format_func=lambda x: "No (< 120 mg/dl)" if x == 0.0 else "Yes (> 120 mg/dl)"
        )
        restecg = st.selectbox(
            "Resting ECG Results:",
            options=[0.0, 1.0, 2.0],
            format_func=lambda x: {
                0.0: "Normal",
                1.0: "ST-T Wave Abnormality",
                2.0: "Left Ventricular Hypertrophy"
            }[x]
        )

    st.subheader("2. Exercise Stress Test Parameters")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        cp = st.selectbox(
            "Chest Pain Type (CP):",
            options=[1.0, 2.0, 3.0, 4.0],
            format_func=lambda x: {
                1.0: "Typical Angina",
                2.0: "Atypical Angina",
                3.0: "Non-Anginal Pain",
                4.0: "Asymptomatic"
            }[x]
        )
        thalach = st.number_input("Maximum Heart Rate Achieved (bpm):", min_value=60, max_value=230, value=150)
        
    with col5:
        exang = st.selectbox(
            "Exercise-Induced Angina:",
            options=[0.0, 1.0],
            format_func=lambda x: "No" if x == 0.0 else "Yes"
        )
        oldpeak = st.slider("ST Depression Induced by Exercise (Oldpeak):", min_value=0.0, max_value=6.5, value=1.0, step=0.1)
        
    with col6:
        slope = st.selectbox(
            "Peak Exercise ST Slope:",
            options=[1.0, 2.0, 3.0],
            format_func=lambda x: {
                1.0: "Upsloping",
                2.0: "Flat",
                3.0: "Downsloping"
            }[x]
        )

    st.subheader("3. Cardiac Imaging & Invasive Diagnostics")
    col7, col8 = st.columns(2)
    
    with col7:
        ca = st.selectbox("Major Vessels Colored by Fluoroscopy (0 - 3):", options=[0.0, 1.0, 2.0, 3.0])
        
    with col8:
        thal = st.selectbox(
            "Thallium Heart Scan (Thal):",
            options=[3.0, 6.0, 7.0],
            format_func=lambda x: {
                3.0: "Normal",
                6.0: "Fixed Defect",
                7.0: "Reversible Defect"
            }[x]
        )
        
    submit_btn = st.form_submit_button("Run Risk Assessment", use_container_width=True)

# Output Section
if submit_btn:
    patient_record = pd.DataFrame([{
        'age': float(age),
        'sex': float(sex),
        'cp': float(cp),
        'trestbps': float(trestbps),
        'chol': float(chol),
        'fbs': float(fbs),
        'restecg': float(restecg),
        'thalach': float(thalach),
        'exang': float(exang),
        'oldpeak': float(oldpeak),
        'slope': float(slope),
        'ca': float(ca),
        'thal': float(thal)
    }])
    
    prediction = pipeline.predict(patient_record)[0]
    probabilities = pipeline.predict_proba(patient_record)[0]
    disease_risk = probabilities[1]
    
    st.divider()
    st.subheader("Assessment Summary")
    
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        if prediction == 1:
            st.error("### 🚨 HIGH RISK: Evidence of Heart Disease")
            st.write(
                "Clinical markers suggest significant likelihood of coronary artery narrowing (≥ 50%). "
                "Further cardiological evaluation (angiography / stress echo) recommended."
            )
        else:
            st.success("### ✅ LOW RISK: No Evidence of Significant Disease")
            st.write(
                "Clinical markers fall within baseline non-critical thresholds. Maintain routine health screenings."
            )
            
    with res_col2:
        st.metric("Estimated Risk Probability", f"{disease_risk * 100:.1f}%")
        st.progress(disease_risk)
        st.caption(f"Confidence (Low Risk: {probabilities[0]*100:.1f}% | High Risk: {probabilities[1]*100:.1f}%)")