import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Load trained model
model = joblib.load(open('claims_fraud_detection.pkl', 'rb')) 

# import os
# model_path = os.path.join(os.getcwd(), 'claims_fraud_detection.pkl')
# model = joblib.load(open(model_path, 'rb'))

# Define Streamlit app
def app():
    
    # Add a header and subheader to the app
    st.title("⚠️ Healthcare Claims Fraud Detection App")
    st.subheader("This app predict whether a claim is fraudulent based on multiple features")

    # Input features with more descriptive names
    st.markdown("### Enter the claim details below and click predict when you are ready:")


    # Input features with more descriptive names
    Provider = st.number_input("Provider's Code")
    InscClaimAmtReimbursed = st.number_input("Insurance Claim Amount Reimbursed")
    IPAnnualReimbursementAmt = st.number_input("Inpatient Annual Reimbursement Amount")
    IPAnnualDeductibleAmt = st.number_input("Inpatient Annual Deductible Amount")
    TotalReimbursement = st.number_input("Total Reimbursement")
    RenalDiseaseIndicator = st.selectbox("Renal Disease Indicator", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    
    # Chronic conditions with user-friendly names
    ChronicCond_Alzheimer = st.selectbox("Alzheimer's Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ChronicCond_Heartfailure = st.selectbox("Heart Failure", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ChronicCond_KidneyDisease = st.selectbox("Kidney Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ChronicCond_Cancer = st.selectbox("Cancer", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ChronicCond_ObstrPulmonary = st.selectbox("Chronic Obstructive Pulmonary Disease (COPD)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ChronicCond_Depression = st.selectbox("Depression", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ChronicCond_Diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ChronicCond_IschemicHeart = st.selectbox("Ischemic Heart Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ChronicCond_Osteoporasis = st.selectbox("Osteoporosis", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ChronicCond_rheumatoidarthritis = st.selectbox("Rheumatoid Arthritis", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ChronicCond_stroke = st.selectbox("Stroke", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    
    # Input data into a DataFrame for model prediction
    if st.button("Predict"):
        input_data = pd.DataFrame([[Provider, InscClaimAmtReimbursed, IPAnnualReimbursementAmt, IPAnnualDeductibleAmt, 
                                    TotalReimbursement, RenalDiseaseIndicator, ChronicCond_Alzheimer, 
                                    ChronicCond_Heartfailure, ChronicCond_KidneyDisease, ChronicCond_Cancer, 
                                    ChronicCond_ObstrPulmonary, ChronicCond_Depression, ChronicCond_Diabetes, 
                                    ChronicCond_IschemicHeart, ChronicCond_Osteoporasis, ChronicCond_rheumatoidarthritis, 
                                    ChronicCond_stroke]], 
                                columns=['Provider', 'InscClaimAmtReimbursed', 'IPAnnualReimbursementAmt', 
                                            'IPAnnualDeductibleAmt', 'TotalReimbursement', 'RenalDiseaseIndicator', 
                                            'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure', 
                                            'ChronicCond_KidneyDisease', 'ChronicCond_Cancer', 
                                            'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression', 
                                            'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart', 
                                            'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis', 
                                            'ChronicCond_stroke'])


        prediction = model.predict(input_data)
        if prediction[0] == 1:
                st.error("🚨 This claim is likely fraudulent!")
        else:
                st.success("✅ This claim seems legitimate.")



if __name__ == "__main__":
    app()


try:
    import joblib
    print("Joblib imported successfully")
except ModuleNotFoundError as e:
    print(f"Joblib import failed: {e}")