import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from tabpfn import TabPFNClassifier

# --- Page Configuration ---
st.set_page_config(
    page_title="AIS Clinical Prediction Tool",
    page_icon="🩺",
    layout="wide"
)

# --- CSS for styling ---
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; color: #0066cc;}
    .sub-header {font-size: 1.5rem; color: #333;}
    .stButton>button {width: 100%; font-weight: bold; height: 3em;}
    </style>
""", unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model():
    # Attempt to load from 'save' folder, fallback to current directory if not found
    path_in_save = os.path.join('save', 'tabpfn_model.pkl')
    path_local = 'tabpfn_model.pkl'
    
    if os.path.exists(path_in_save):
        return joblib.load(path_in_save)
    elif os.path.exists(path_local):
        return joblib.load(path_local)
    else:
        st.error("Error: 'tabpfn_model.pkl' not found in 'save/' folder or root directory.")
        return None

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model. Please check dependencies (torch, tabpfn). Error: {e}")
    model = None

# --- Main Interface ---
st.markdown('<p class="main-header">AIS Clinical Prediction Model</p>', unsafe_allow_html=True)
st.markdown("This tool uses **TabPFN** to predict the clinical outcome based on patient characteristics.")
st.markdown("---")

# --- Input Form ---
st.sidebar.header("📋 Patient Data Input")
st.sidebar.markdown("Please enter the patient's clinical parameters below:")

# Creating a form for cleaner UI
with st.form("prediction_form"):
    
    st.subheader("1. Continuous Variables")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # ONT: Onset to Needle Time? Assuming minutes. Adjust label/help if needed.
        ont = st.number_input("ONT", min_value=0.0, value=60.0, step=1.0, help="Onset to Needle Time")
        # Baseline NIHSS: Range usually 0-42
        b_nihss = st.number_input("Baseline NIHSS", min_value=0.0, max_value=42.0, value=10.0, step=1.0)
        # Glucose: Assuming mmol/L or mg/dL
        glucose = st.number_input("Glucose", min_value=0.0, value=5.5, step=0.1)

    with col2:
        # SBP: Systolic Blood Pressure
        sbp = st.number_input("SBP (mmHg)", min_value=0.0, max_value=300.0, value=140.0, step=1.0)
        # Post-thrombolysis NIHSS
        p_nihss = st.number_input("Post-thrombolysis NIHSS", min_value=0.0, max_value=42.0, value=8.0, step=1.0)
        # PT: Prothrombin Time
        pt = st.number_input("PT (s)", min_value=0.0, value=12.0, step=0.1)

    with col3:
        # BMI: Body Mass Index
        bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=24.0, step=0.1)
        # Neutrophil Ratio: Usually 0.0-1.0 or 0-100
        neu_ratio = st.number_input("Neutrophil Ratio", min_value=0.0, value=0.6, step=0.01)
        # TT: Thrombin Time
        tt = st.number_input("TT (s)", min_value=0.0, value=16.0, step=0.1)

    st.subheader("2. Binary Variables")
    col_b1, col_b2, col_b3 = st.columns(3)
    
    # Function to map Yes/No to 1/0
    def map_binary(val):
        return 1 if val == "Yes" else 0

    with col_b1:
        htn_input = st.selectbox("Hypertension", options=["No", "Yes"])
        htn = map_binary(htn_input)
    
    with col_b2:
        stroke_input = st.selectbox("History of Stroke", options=["No", "Yes"])
        stroke = map_binary(stroke_input)
        
    with col_b3:
        smoking_input = st.selectbox("Smoking", options=["No", "Yes"])
        smoking = map_binary(smoking_input)

    # Submit Button
    submitted = st.form_submit_button("Run Prediction 🚀")

# --- Prediction Logic ---
if submitted:
    if model is None:
        st.error("Model is not loaded.")
    else:
        # Construct DataFrame with EXACT column names used in training
        # Features: ONT, SBP, BMI, Baseline NIHSS, Post-thrombolysis NIHSS, Neutrophil Ratio, Glucose, PT, TT, Hypertension, Stroke, Smoking
        
        input_data = pd.DataFrame({
            'ONT': [ont],
            'SBP': [sbp],
            'BMI': [bmi],
            'Baseline NIHSS': [b_nihss],
            'Post-thrombolysis NIHSS': [p_nihss],
            'Neutrophil Ratio': [neu_ratio],
            'Glucose': [glucose],
            'PT': [pt],
            'TT': [tt],
            'Hypertension': [htn],
            'Stroke': [stroke],
            'Smoking': [smoking]
        })

        # Ensure column order matches training (Dicts maintain order in Python 3.7+, but safe to enforce)
        feature_order = ['ONT', 'SBP', 'BMI', 'Baseline NIHSS', 'Post-thrombolysis NIHSS', 
                         'Neutrophil Ratio', 'Glucose', 'PT', 'TT', 'Hypertension', 'Stroke', 'Smoking']
        input_data = input_data[feature_order]

        # Display Input Data Summary
        with st.expander("View Input Data"):
            st.dataframe(input_data)

        # Predict
        with st.spinner("Analyzing clinical data with TabPFN..."):
            try:
                # Get Probability
                # TabPFN returns [prob_class_0, prob_class_1]
                probs = model.predict_proba(input_data)
                prob_pos = probs[0][1] # Probability of the positive class (e.g., poor outcome/disease)
                
                # Get Label
                pred_label = model.predict(input_data)[0]

                # --- Results Display ---
                st.markdown("### Prediction Results")
                
                r_col1, r_col2 = st.columns([1, 2])
                
                with r_col1:
                    if pred_label == 1:
                        st.error(f"**Class: Positive (1)**")
                    else:
                        st.success(f"**Class: Negative (0)**")
                
                with r_col2:
                    st.metric(label="Probability of Positive Outcome", value=f"{prob_pos:.2%}")
                    st.progress(int(prob_pos * 100))

                # Interpretation text (Optional)
                if prob_pos > 0.5:
                    st.warning("⚠️ High Probability: The model predicts a positive outcome (e.g., event occurrence).")
                else:
                    st.info("✅ Low Probability: The model predicts a negative outcome.")

            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.write("Please check if the input data types match the training data.")

# --- Footer / Disclaimer ---
st.markdown("---")
st.caption("⚠️ **Disclaimer:** This model is for research and educational purposes only. It should not be used as the sole basis for clinical diagnosis or treatment decisions.")