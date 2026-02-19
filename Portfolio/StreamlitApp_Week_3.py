import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

import shap

# Setup & Path Configuration
warnings.simplefilter("ignore")

# Fix path for Streamlit Cloud (ensure 'src' is findable)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import extract_features

# Access the secrets
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# AWS Session Management
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name="us-east-1"
    )

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# IMPORTANT: compute features only on demand (prevents timeout on every rerun)
@st.cache_data(ttl=3600)
def get_features():
    return extract_features()

MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": "explainer.shap",
    "pipeline": "finalized_model.tar.gz",
    # MUST match training feature order
    "keys": ["SBUX", "NKE", "CMG", "DEXJPUS", "DEXUSUK", "SP500", "DJIA", "VIXCLS"],
    "inputs": [
        {"name": k, "type": "number", "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01}
        for k in ["SBUX", "NKE", "CMG", "DEXJPUS", "DEXUSUK", "SP500", "DJIA", "VIXCLS"]
    ]
}

def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client("s3")
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)

def call_model_api(input_df: pd.DataFrame):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer()
    )

    try:
        payload = input_df.to_numpy(dtype=np.float32)
        raw_pred = predictor.predict(payload)
        pred_val = float(np.asarray(raw_pred).ravel()[-1])
        return round(pred_val, 4), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        session,
        aws_bucket,
        posixpath.join("explainer", explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name),
    )

    # Explain the LAST row only (your new input)
    shap_values = explainer(input_df.iloc[[-1]])

   def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        session,
        aws_bucket,
        posixpath.join("explainer", explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name),
    )

    # Explain the LAST row only (your new input)
    shap_values = explainer(input_df.iloc[[-1]])

    st.subheader("🔍 Decision Transparency (SHAP)")

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig, clear_figure=True)

    # Pick the most influential feature by absolute SHAP value
    vals = shap_values[0].values
    idx = int(np.argmax(np.abs(vals)))
    top_feature = shap_values[0].feature_names[idx]

    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")



# Streamlit UI
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 ML Deployment")

with st.form("pred_form"):
    st.subheader("Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                inp["name"].replace("_", " ").upper(),
                min_value=inp["min"],
                max_value=inp["max"],
                value=inp["default"],
                step=inp["step"]
            )

    submitted = st.form_submit_button("Run Prediction")

if submitted:
    data_row = [user_inputs[k] for k in MODEL_INFO["keys"]]

    with st.spinner("Fetching features (FRED/YFinance)..."):
        base_df = get_features()

    # Make sure column order matches the model
    base_df = base_df[MODEL_INFO["keys"]]

    # Append your custom row to the end
    input_df = pd.concat([base_df, pd.DataFrame([data_row], columns=MODEL_INFO["keys"])])

    res, status = call_model_api(input_df)
    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(input_df, session, aws_bucket)
    else:
        st.error(res)







