# views/fraud_detection.py
# Complete, safe, and deploy-friendly Streamlit view for fraud detection.
# - Lazy-loads and caches the model
# - Handles missing model file gracefully
# - Can optionally download model from a URL stored in Streamlit secrets (MODEL_URL)
# - Shows friendly messages instead of crashing
# - Uses the same input fields you had, with validation & helpful outputs

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import traceback
from io import BytesIO

# Optional: requests used only if we download model from a URL
try:
    import requests
except Exception:
    requests = None

# Path (relative to repo root) where you keep the model file
MODEL_PATH = "claims_fraud_detection.pkl"

# Helper: load model from local path or from URL in st.secrets["MODEL_URL"]
@st.cache_resource  # caches across runs/sessions so model is loaded only once
def get_model():
    """
    Attempt to load the model from disk. If not found, and if a MODEL_URL
    is present in st.secrets, attempt to download the model and load it.
    Returns: model object or None on failure.
    """
    # 1) Local file exists?
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            st.session_state["_model_load_msg"] = "Loaded model from local file."
            return model
        except Exception as e:
            # Log full traceback to stdout (appears in Streamlit logs)
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            print("=== MODEL LOAD ERROR (local) ===")
            print(tb)
            print("=== END MODEL LOAD ERROR ===")
            st.session_state["_model_load_msg"] = "Failed to load model from local file."
            return None

    # 2) Try downloading from a URL if provided in secrets
    model_url = st.secrets.get("MODEL_URL") if st.secrets is not None else None
    if model_url:
        if requests is None:
            st.session_state["_model_load_msg"] = "requests is not available to download model."
            return None
        try:
            resp = requests.get(model_url, timeout=20)
            resp.raise_for_status()
            buf = BytesIO(resp.content)
            model = joblib.load(buf)
            st.session_state["_model_load_msg"] = "Downloaded and loaded model from MODEL_URL secret."
            return model
        except Exception as e:
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            print("=== MODEL LOAD ERROR (download) ===")
            print(tb)
            print("=== END MODEL LOAD ERROR ===")
            st.session_state["_model_load_msg"] = "Failed to download or load model from MODEL_URL."
            return None

    # 3) Nothing we can do
    st.session_state["_model_load_msg"] = (
        f"Model file not found at '{MODEL_PATH}'. "
        "If you want the app to make predictions, either upload the model file to the repo "
        "or set a MODEL_URL in Streamlit secrets pointing to a raw .pkl file."
    )
    return None


def build_input_dataframe():
    """
    Presents input widgets and returns a DataFrame with a single row matching the model's expected columns.
    Edit column names/order if your model expects different feature names.
    """
    st.markdown("### Enter the claim details below and click **Predict** when ready:")

    Provider = st.number_input("Provider's Code", value=0, step=1)
    InscClaimAmtReimbursed = st.number_input("Insurance Claim Amount Reimbursed", value=0.0, step=1.0, format="%.2f")
    IPAnnualReimbursementAmt = st.number_input("Inpatient Annual Reimbursement Amount", value=0.0, step=1.0, format="%.2f")
    IPAnnualDeductibleAmt = st.number_input("Inpatient Annual Deductible Amount", value=0.0, step=1.0, format="%.2f")
    TotalReimbursement = st.number_input("Total Reimbursement", value=0.0, step=1.0, format="%.2f")

    RenalDiseaseIndicator = st.selectbox("Renal Disease Indicator", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    ChronicCond_Alzheimer = st.selectbox("Alzheimer's Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ChronicCond_Heartfailure = st.selectbox("Heart Failure", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ChronicCond_KidneyDisease = st.selectbox("Kidney Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ChronicCond_Cancer = st.selectbox("Cancer", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ChronicCond_ObstrPulmonary = st.selectbox("COPD (Chronic Obstructive Pulmonary Disease)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ChronicCond_Depression = st.selectbox("Depression", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ChronicCond_Diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ChronicCond_IschemicHeart = st.selectbox("Ischemic Heart Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ChronicCond_Osteoporasis = st.selectbox("Osteoporosis", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ChronicCond_rheumatoidarthritis = st.selectbox("Rheumatoid Arthritis", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ChronicCond_stroke = st.selectbox("Stroke", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    # Build DataFrame
    input_data = pd.DataFrame([[
        Provider, InscClaimAmtReimbursed, IPAnnualReimbursementAmt,
        IPAnnualDeductibleAmt, TotalReimbursement, RenalDiseaseIndicator,
        ChronicCond_Alzheimer, ChronicCond_Heartfailure, ChronicCond_KidneyDisease,
        ChronicCond_Cancer, ChronicCond_ObstrPulmonary, ChronicCond_Depression,
        ChronicCond_Diabetes, ChronicCond_IschemicHeart, ChronicCond_Osteoporasis,
        ChronicCond_rheumatoidarthritis, ChronicCond_stroke
    ]], columns=[
        'Provider', 'InscClaimAmtReimbursed', 'IPAnnualReimbursementAmt',
        'IPAnnualDeductibleAmt', 'TotalReimbursement', 'RenalDiseaseIndicator',
        'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure',
        'ChronicCond_KidneyDisease', 'ChronicCond_Cancer',
        'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression',
        'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart',
        'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis',
        'ChronicCond_stroke'
    ])

    return input_data


def app():
    """
    Main view entrypoint used by streamlit_app.py:
      from views import fraud_detection
      fraud_detection.app()
    """
    st.title("⚠️ Healthcare Claims Fraud Detection")
    st.write("This demo predicts whether a healthcare claim looks fraudulent. "
             "If predictions are unavailable it means the trained model couldn't be loaded in this environment.")

    # Show any model load message from cache
    _ = get_model()  # ensure model load attempted & cached message populated
    load_msg = st.session_state.get("_model_load_msg")
    if load_msg:
        st.info(load_msg)

    # Build input UI
    input_data = build_input_dataframe()

    # Predict when user clicks
    if st.button("Predict"):
        model = get_model()
        if model is None:
            st.warning("Prediction unavailable: model not loaded. Check logs or upload the model file to the repo.")
            # Helpful instructions for the user / developer
            st.markdown(
                "- Ensure `claims_fraud_detection.pkl` is present in the repo root, **or**\n"
                "- Add a raw download link to the model as `MODEL_URL` in Streamlit app Secrets (Manage app → Settings → Secrets)."
            )
            return

        # Validate that model implements predict
        if not hasattr(model, "predict"):
            st.error("Loaded object does not appear to be a model with a `predict()` method.")
            return

        try:
            # Some models expect a specific dtype/order; convert numeric columns to float
            input_data_numeric = input_data.astype(float)
            prediction = model.predict(input_data_numeric)

            # If model supports predict_proba, show probability for the positive class (if applicable)
            prob = None
            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(input_data_numeric)
                    # assume positive class is index 1 if two-class
                    if probs.shape[1] >= 2:
                        prob = probs[:, 1][0]
                except Exception:
                    prob = None  # ignore probability failures

            # Show results
            if prediction is None:
                st.error("Model returned no prediction.")
                return

            pred_label = int(np.asarray(prediction)[0])
            if pred_label == 1:
                st.error("🚨 This claim is likely fraudulent!")
                if prob is not None:
                    st.write(f"Estimated fraud probability: **{prob:.2%}**")
            else:
                st.success("✅ This claim seems legitimate.")
                if prob is not None:
                    st.write(f"Estimated fraud probability: **{prob:.2%}**")

            # Show the input back to the user for verification (optional)
            with st.expander("Show input data"):
                st.write(input_data)
        except Exception as e:
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            print("=== PREDICTION ERROR ===")
            print(tb)
            print("=== END PREDICTION ERROR ===")
            st.error("An error occurred during prediction. Check logs for details.")

# If this file is executed directly (for debugging), run the app
if __name__ == "__main__":
    app()
