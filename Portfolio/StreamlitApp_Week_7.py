import json
import boto3
import pandas as pd
import streamlit as st

st.set_page_config(page_title="BTC Signal Predictor", layout="centered")
st.title("Bitcoin Signal Prediction (SageMaker Endpoint)")

FEATURE_COLS = ["EMA_10", "ROC_10", "MOM_10", "RSI_10", "MA_10"]
LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}

# --- Secrets loader (works with or without [aws_credentials]) ---
secrets = st.secrets["aws_credentials"] if "aws_credentials" in st.secrets else st.secrets

region = secrets.get("AWS_REGION", "us-east-1")  # default if missing
endpoint = secrets.get("SAGEMAKER_ENDPOINT_NAME")

required = ["SAGEMAKER_ENDPOINT_NAME", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
missing = [k for k in required if not secrets.get(k)]
if missing:
    st.error(f"Missing secrets: {missing}")
    st.write("Available secret keys:", list(secrets.keys()))
    st.stop()

session = boto3.Session(
    aws_access_key_id=secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=secrets["AWS_SECRET_ACCESS_KEY"],
    aws_session_token=secrets.get("AWS_SESSION_TOKEN"),
    region_name=region,
)

runtime = session.client("sagemaker-runtime", region_name=region)

st.subheader("Enter feature values")
c1, c2 = st.columns(2)

with c1:
    ema = st.number_input("EMA_10", value=0.0, format="%.6f")
    roc = st.number_input("ROC_10", value=0.0, format="%.6f")
    mom = st.number_input("MOM_10", value=0.0, format="%.6f")
with c2:
    rsi = st.number_input("RSI_10", value=50.0, format="%.6f")
    ma  = st.number_input("MA_10", value=0.0, format="%.6f")

if st.button("Predict Signal"):
    df = pd.DataFrame([[ema, roc, mom, rsi, ma]], columns=FEATURE_COLS)

    # Send CSV WITH HEADER so inference can read column names
    body = df.to_csv(index=False).encode("utf-8")

    try:
        resp = runtime.invoke_endpoint(
            EndpointName=endpoint,
            ContentType="text/csv",
            Accept="application/json",
            Body=body,
        )
        pred = json.loads(resp["Body"].read().decode("utf-8"))

        pred_int = int(pred[0]) if isinstance(pred, list) else int(pred)
        st.success(f"Prediction: {LABEL_MAP.get(pred_int, pred_int)} ({pred_int})")
        st.write("Raw response:", pred)

    except Exception as e:
        st.error(f"Invoke failed: {e}")






