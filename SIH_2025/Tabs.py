import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
import base64
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from Model import predict_risk   # external ML model


# ==============================
# Utility Functions
# ==============================
def send_to_backend(uploaded_file):
    """Send uploaded file to backend server"""
    if uploaded_file is None:
        return None
    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    try:
        response = requests.post("http://127.0.0.1:5000/upload", files=files, timeout=10)
        if response.status_code == 200:
            st.success(f"✅ {uploaded_file.name} saved to backend")
            return response.json().get("path")
        else:
            st.error(f"❌ Backend upload failed ({response.status_code})")
            return None
    except Exception as e:
        st.error(f"Backend error: {e}")
        return None


def compute_factors(row, config):
    """Compute factors dynamically from row + config thresholds"""
    # Attendance
    att = float(row.get("attendance_percent", 1.0) or 1.0)
    att_threshold = config.get("att_threshold", 0.75)
    att_factor = max(0.0, (att_threshold - att) / (att_threshold or 1.0))

    # Assessment
    avg = float(row.get("avg_score", 1.0) or 1.0)
    score_threshold = config.get("score_threshold", 0.4)
    ass_factor = max(0.0, (score_threshold - avg) / (score_threshold or 1.0))

    # Fees
    fee_paid = row.get("fees_paid_ratio", None)
    frac_unpaid = 1.0 - float(fee_paid) if fee_paid is not None else row.get("fraction_unpaid", 0.0)
    fee_factor = max(0.0, float(frac_unpaid))

    # Attempts
    attempts = float(row.get("attempts_used", 0.0) or 0.0)
    max_att = config.get("max_attempts", 5) or 5.0
    attempts_factor = min(attempts / max_att, 1.0)

    return att_factor, ass_factor, fee_factor, attempts_factor


def compute_risk(row, config):
    """Compute weighted risk score and assign flag"""
    att_factor, ass_factor, fee_factor, attempts_factor = compute_factors(row, config)
    risk = (config.get("w_att", 0.0) * att_factor +
            config.get("w_ass", 0.0) * ass_factor +
            config.get("w_fee", 0.0) * fee_factor +
            config.get("w_attempts", 0.0) * attempts_factor)

    thresholds = config.get("risk_thresholds", {"low": 0.4, "medium": 0.7})
    if risk < thresholds["low"]:
        flag = "🟢"
    elif risk < thresholds["medium"]:
        flag = "🟡"
    else:
        flag = "🔴"

    return risk, flag, att_factor, ass_factor, fee_factor, attempts_factor


def preprocess_data(att, ass, fees, config, model_results=None):
    """Merge & preprocess student data"""
    if att is None or ass is None or fees is None:
        return None

    # Attendance
    if "attendance_percent" not in att.columns and "attendance" in att.columns:
        att = att.rename(columns={"attendance": "attendance_percent"})
    att_agg = att.groupby("student_id").agg({"attendance_percent": "mean", "student_name": "first"}).reset_index()

    # Assessment
    if "avg_score" not in ass.columns and "score_percent" in ass.columns:
        ass = ass.rename(columns={"score_percent": "avg_score"})
    ass_agg = ass[["student_id", "avg_score"]].copy()

    # Fees
    if "fees_paid_ratio" not in fees.columns and "paid_ratio" in fees.columns:
        fees = fees.rename(columns={"paid_ratio": "fees_paid_ratio"})
    fees_agg = fees[["student_id", "fees_paid_ratio"]].copy()
    fees_agg["fraction_unpaid"] = 1 - fees_agg["fees_paid_ratio"].fillna(0.0)

    # Merge
    df = att_agg.merge(ass_agg, on="student_id", how="left").merge(fees_agg, on="student_id", how="left")
    df = df.fillna({"attendance_percent": 1.0, "avg_score": 1.0, "fraction_unpaid": 0.0, "attempts_used": 0})

    # Attempts
    if "num_failed_attempts" in ass.columns:
        df["attempts_used"] = df.get("num_failed_attempts", 0)

    # Apply ML results if available
    if model_results is not None and "student_id" in model_results.columns:
        df = df.merge(model_results[["student_id", "final_dropout_probability"]], on="student_id", how="left")
        df["ml_probability"] = df["final_dropout_probability"]
        df["ml_flag"] = df["ml_probability"].apply(lambda p: "🟢" if p < 0.4 else ("🟡" if p < 0.7 else "🔴"))

    # Always compute rule-based risk
    risk_results = df.apply(lambda r: compute_risk(r, config), axis=1)
    df[["rule_risk_score", "rule_flag", "att_factor", "ass_factor", "fee_factor", "attempts_factor"]] = pd.DataFrame(
        risk_results.tolist(), index=df.index
    )

    return df


def send_email_report(to_email, student_data):
    """Send email report securely"""
    sender_email = "vaibhavraikwar505@gmail.com"
    sender_password = os.getenv("EMAIL_PASS")  # <-- use environment variable
    if not sender_password:
        st.error("⚠️ EMAIL_PASS not set in environment.")
        return False

    subject = f"Dropout Risk Alert: {student_data.get('student_name','Unknown')}"
    body = f"""
    Student: {student_data.get('student_name','Unknown')}
    Risk Score: {student_data.get('rule_risk_score',0):.2f}
    Risk Level: {student_data.get('rule_flag','N/A')}
    """

    msg = MIMEMultipart()
    msg["From"], msg["To"], msg["Subject"] = sender_email, to_email, subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Email send error: {e}")
        return False


# ==============================
# Streamlit App UI
# ==============================
st.set_page_config(page_title="MetaMentor", layout="wide")
st.title("AI-based Dropout Prediction & Counseling System")

# Sidebar Config
st.sidebar.header("Configuration")
att_threshold = st.sidebar.slider("Attendance Threshold", 0.0, 1.0, 0.75)
score_threshold = st.sidebar.slider("Score Threshold", 0.0, 1.0, 0.4)
fee_overdue_days = st.sidebar.number_input("Fee Overdue Days Threshold", 0, 180, 30)
max_attempts = st.sidebar.number_input("Max Exam Attempts Allowed", 1, 10, 5)
w_att = st.sidebar.slider("Weight: Attendance", 0.0, 1.0, 0.3)
w_ass = st.sidebar.slider("Weight: Assessment", 0.0, 1.0, 0.4)
w_fee = st.sidebar.slider("Weight: Fee", 0.0, 1.0, 0.2)
w_attempts = st.sidebar.slider("Weight: Attempts", 0.0, 1.0, 0.1)

config = {
    "att_threshold": att_threshold,
    "score_threshold": score_threshold,
    "fee_overdue_days": fee_overdue_days,
    "max_attempts": max_attempts,
    "w_att": w_att,
    "w_ass": w_ass,
    "w_fee": w_fee,
    "w_attempts": w_attempts,
    "risk_thresholds": {"low": 0.4, "medium": 0.7},
}

# Tabs instead of Selectbox
tab1, tab2, tab3 = st.tabs(["📂 Upload", "📊 Performance Table", "📈 Dashboard"])

with tab1:
    st.subheader("Upload Student Data")
    for label, key in [("Attendance", "att_df"), ("Assessment", "ass_df"), ("Fees", "fees_df")]:
        file = st.file_uploader(f"Upload {label} CSV", type=["csv"], key=f"{key}_file")
        if file:
            st.session_state[key] = pd.read_csv(file)
            send_to_backend(file)

with tab2:
    att, ass, fees = st.session_state.get("att_df"), st.session_state.get("ass_df"), st.session_state.get("fees_df")
    try:
        model_results = predict_risk()
    except Exception:
        model_results = None

    df = preprocess_data(att, ass, fees, config, model_results)
    if df is not None:
        st.dataframe(df.head(), use_container_width=True)
    else:
        st.info("Please upload all required files.")

with tab3:
    att, ass, fees = st.session_state.get("att_df"), st.session_state.get("ass_df"), st.session_state.get("fees_df")
    try:
        model_results = predict_risk()
    except Exception:
        model_results = None

    df = preprocess_data(att, ass, fees, config, model_results)
    if df is not None:
        st.metric("Total Students", len(df))
        st.write(df[["student_id", "student_name", "rule_risk_score", "rule_flag"]].head())
    else:
        st.info("Please upload all required files.")
