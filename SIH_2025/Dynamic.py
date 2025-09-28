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
from Model import predict_risk

# ---------------------------
# Helper functions & config
# ---------------------------

def send_to_backend(uploaded_file):
    """Send uploaded file bytes to local backend (if available)."""
    if uploaded_file is not None:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        try:
            response = requests.post("http://127.0.0.1:5000/upload", files=files, timeout=8)
            if response.status_code == 200:
                st.success("✅ File saved to backend")
                return response.json().get("path")
            else:
                st.warning("Backend upload returned status: %s" % response.status_code)
                return None
        except Exception as e:
            st.info(f"Could not reach backend: {e}")
            return None
    return None

def compute_factors(row, config):
    """Return att_factor, ass_factor, fee_factor, attempts_factor"""
    att = row.get('attendance_percent', row.get('attendance', 1.0))
    try:
        att = float(att)
    except Exception:
        att = 1.0
    att_threshold = config.get('att_threshold', 0.75) or 0.75
    att_factor = max(0.0, (att_threshold - att) / (att_threshold if att_threshold != 0 else 1.0))

    avg = row.get('avg_score', row.get('score_percent', 1.0))
    try:
        avg = float(avg)
    except Exception:
        avg = 1.0
    score_threshold = config.get('score_threshold', 0.4) or 0.4
    ass_factor = max(0.0, (score_threshold - avg) / (score_threshold if score_threshold != 0 else 1.0))

    fee_paid = row.get('fees_paid_ratio', None)
    if fee_paid is None or (isinstance(fee_paid, float) and np.isnan(fee_paid)):
        frac_unpaid = row.get('fraction_unpaid', 0.0)
    else:
        try:
            frac_unpaid = 1.0 - float(fee_paid)
        except Exception:
            frac_unpaid = row.get('fraction_unpaid', 0.0)
    try:
        fee_factor = max(0.0, float(frac_unpaid))
    except Exception:
        fee_factor = 0.0

    attempts = row.get('attempts_used', row.get('num_failed_attempts', 0))
    try:
        attempts = float(attempts)
    except Exception:
        attempts = 0.0
    max_att = config.get('max_attempts', 5)
    if max_att is None or max_att == 0:
        max_att = 5.0
    attempts_factor = min(attempts / float(max_att), 1.0)

    return att_factor, ass_factor, fee_factor, attempts_factor

def compute_risk_from_factors(att_factor, ass_factor, fee_factor, attempts_factor, config):
    """Compute a simple linear combination risk score and flag"""
    risk = (config.get('w_att', 0.0) * att_factor +
            config.get('w_ass', 0.0) * ass_factor +
            config.get('w_fee', 0.0) * fee_factor +
            config.get('w_attempts', 0.0) * attempts_factor)
    if risk < 0.4:
        flag = "🟢"
    elif risk < 0.7:
        flag = "🟡"
    else:
        flag = "🔴"
    return risk, flag

@st.cache_data
def prepare_risk_dataframe(att_df, ass_df, fees_df, model_results, config):
    """
    Merge inputs, compute features, incorporate model_results (if provided),
    and compute risk metrics.
    Returns a DataFrame ready for display.
    """
    # defensive copies
    att = att_df.copy()
    ass = ass_df.copy()
    fees = fees_df.copy()

    # normalize column names
    if 'attendance_percent' not in att.columns and 'attendance' in att.columns:
        att = att.rename(columns={'attendance': 'attendance_percent'})

    if 'avg_score' not in ass.columns and 'score_percent' in ass.columns:
        ass = ass.rename(columns={'score_percent': 'avg_score'})

    if 'fees_paid_ratio' not in fees.columns and 'paid_ratio' in fees.columns:
        fees = fees.rename(columns={'paid_ratio': 'fees_paid_ratio'})

    # aggregate attendance: mean per student
    att_agg = att.groupby('student_id').agg({
        'attendance_percent': 'mean',
        'student_name': 'first'
    }).reset_index()

    ass_agg = ass[['student_id', 'avg_score']].copy()

    fees_agg = fees[['student_id', 'fees_paid_ratio']].copy()
    fees_agg['fraction_unpaid'] = 1 - fees_agg['fees_paid_ratio'].fillna(0.0)

    df = att_agg.merge(ass_agg, on="student_id", how="left").merge(fees_agg, on="student_id", how="left")

    # sensible defaults (change if you want different behavior)
    df = df.fillna({'attendance_percent': 1.0, 'avg_score': 1.0, 'fraction_unpaid': 0.0, 'attempts_used': 0})

    # attempts column handling
    if 'num_failed_attempts' in ass.columns:
        df['attempts_used'] = df.get('num_failed_attempts', 0)
    else:
        df['attempts_used'] = df.get('attempts_used', 0)

    # incorporate model predictions if available
    if model_results is not None and isinstance(model_results, pd.DataFrame) and 'student_id' in model_results.columns:
        merged = df.merge(model_results[['student_id', 'final_dropout_probability']], on='student_id', how='left')
        df = merged
        df['ml_probability'] = df['final_dropout_probability']
        # Use ML-provided probability as main risk_score when present; fallback to computed.
        df['risk_score'] = df['final_dropout_probability'].fillna(np.nan)
    else:
        df['ml_probability'] = np.nan
        df['risk_score'] = np.nan

    # compute factors row-wise (vectorization would be possible but keep readability)
    factors = df.apply(lambda r: compute_factors(r, config), axis=1)
    df[['att_factor','ass_factor','fee_factor','attempts_factor']] = pd.DataFrame(factors.tolist(), index=df.index)

    # if ml_probability missing -> compute risk_score from factors; otherwise keep ml-prob
    computed = df.apply(
        lambda r: compute_risk_from_factors(r['att_factor'], r['ass_factor'], r['fee_factor'], r['attempts_factor'], config),
        axis=1
    )
    df[['computed_risk_score','computed_risk_flag']] = pd.DataFrame(computed.tolist(), index=df.index)

    # final risk_score: use ML probability when available, else computed
    df['risk_score'] = df['risk_score'].fillna(df['computed_risk_score'])
    df['risk_flag'] = df.apply(lambda r: ("🟢" if r['risk_score'] < 0.4 else ("🟡" if r['risk_score'] < 0.7 else "🔴")), axis=1)

    # keep consistent column order
    ordered_cols = ['student_id','student_name','attendance_percent','avg_score','fees_paid_ratio','fraction_unpaid','attempts_used',
                    'att_factor','ass_factor','fee_factor','attempts_factor','ml_probability','computed_risk_score','risk_score','risk_flag']
    for c in ordered_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df[ordered_cols]

def send_email_report(to_email, student_data):
    """
    Send a plain text email using credentials stored in st.secrets.
    Requires .streamlit/secrets.toml with [gmail] sender_email and app_password
    """
    try:
        sender_email = st.secrets["gmail"]["sender_email"]
        sender_password = st.secrets["gmail"]["app_password"]
    except Exception:
        st.error("Email credentials not configured in st.secrets; cannot send email.")
        return False

    subject = f"Dropout Risk Alert: {student_data.get('student_name','Unknown')}"
    body = f"""
Dear Mentor/Guardian,

Student: {student_data.get('student_name','Unknown')}
Risk Score: {student_data.get('risk_score',0):.2f}
Risk Level: {student_data.get('risk_flag','N/A')}

Key Factors:
- Attendance Factor: {student_data.get('att_factor',0):.2f}
- Assessment Factor: {student_data.get('ass_factor',0):.2f}
- Fee Factor: {student_data.get('fee_factor',0):.2f}
- Attempts Factor: {student_data.get('attempts_factor',0):.2f}

Please consider early intervention.

Regards,
Dropout Prediction System
"""
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Email send error: {e}")
        return False

# ---------------------------
# UI & App code
# ---------------------------

# background image (fixed data URI)
image_path = os.path.join("images", "BG.jpg")
try:
    with open(image_path, "rb") as f:
        encoded_img = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_img}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
except Exception as e:
    # print to console; don't crash the app if background missing
    print("Background image could not be set:", e)

st.set_page_config(page_title="MetaMentor", layout="wide")
st.title("AI-based Dropout Prediction & Counseling System")

st.sidebar.header("Configuration")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #002d00;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("**➡️ Set Threshold**")
att_threshold = st.sidebar.slider("Attendance Threshold", 0.0, 1.0, 0.75)
score_threshold = st.sidebar.slider("Score Threshold", 0.0, 1.0, 0.4)
fee_overdue_days = st.sidebar.number_input("Fee Overdue Days Threshold", 0, 180, 30)
max_attempts = st.sidebar.number_input("Max Exam Attempts Allowed", 1, 10, 5)

st.sidebar.markdown("**➡️ Set Weightage**")
w_att = st.sidebar.slider("Weight: Attendance", 0.0, 1.0, 0.3)
w_ass = st.sidebar.slider("Weight: Assessment", 0.0, 1.0, 0.4)
w_fee = st.sidebar.slider("Weight: Fee", 0.0, 1.0, 0.2)
w_attempts = st.sidebar.slider("Weight: Attempts", 0.0, 1.0, 0.1)

config = {
    'att_threshold': att_threshold,
    'score_threshold': score_threshold,
    'fee_overdue_days': fee_overdue_days,
    'max_attempts': max_attempts,
    'w_att': w_att,
    'w_ass': w_ass,
    'w_fee': w_fee,
    'w_attempts': w_attempts
}

option = st.selectbox("", ["Upload File", "Performance Table", "Dashboard"], index=0)

# ---------------------------
# Upload page
# ---------------------------
if option == "Upload File":
    st.subheader("📂 Upload Student Data")
    uploaded_att = st.file_uploader("Upload Attendance CSV", type=["csv"], key="att_file")
    if uploaded_att is not None:
        st.session_state['att_df'] = pd.read_csv(uploaded_att)
        backend_path = send_to_backend(uploaded_att)
        if backend_path:
            st.write("Saved at:", backend_path)

    uploaded_ass = st.file_uploader("Upload Assessment CSV", type=["csv"], key="ass_file")
    if uploaded_ass is not None:
        st.session_state['ass_df'] = pd.read_csv(uploaded_ass)
        backend_path = send_to_backend(uploaded_ass)
        if backend_path:
            st.write("Saved at:", backend_path)

    uploaded_fees = st.file_uploader("Upload Fees CSV", type=["csv"], key="fees_file")
    if uploaded_fees is not None:
        st.session_state['fees_df'] = pd.read_csv(uploaded_fees)
        backend_path = send_to_backend(uploaded_fees)
        if backend_path:
            st.write("Saved at:", backend_path)

# ---------------------------
# Performance Table
# ---------------------------
elif option == "Performance Table":
    att = st.session_state.get('att_df')
    ass = st.session_state.get('ass_df')
    fees = st.session_state.get('fees_df')

    if att is not None and ass is not None and fees is not None:
        st.title("🎓 Dropout Risk Prediction Dashboard")
        st.write("⚙ Running model...")

        # attempt to call model (if available). Keep it optional.
        try:
            model_results = predict_risk()
            if not (isinstance(model_results, pd.DataFrame) and 'final_dropout_probability' in model_results.columns):
                model_results = None
        except Exception as e:
            model_results = None

        df = prepare_risk_dataframe(att, ass, fees, model_results, config)

        st.success("✅ Model processed successfully! Here are the at-risk students:")
        st.subheader("📃 All Students Risk Data")
        display_cols = ['student_id','student_name','attendance_percent','avg_score','risk_score','risk_flag','ml_probability']
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[display_cols], hide_index=True)
        st.markdown("___")

        st.subheader("👥 Filter Students")
        risk_count = df['risk_flag'].value_counts() if 'risk_flag' in df.columns else pd.Series(dtype=int)
        label_map = {"Low Risk": "🟢", "Medium Risk": "🟡", "High Risk": "🔴"}
        risk_level = st.selectbox("Select Risk Level", ["All"] + list(label_map.keys()))
        if risk_level != "All":
            emoji_value = label_map[risk_level]
            filtered = df[df['risk_flag'] == emoji_value]
        else:
            filtered = df

        display_cols_filtered = [c for c in ['student_id','student_name','attendance_percent','avg_score','risk_score','risk_flag','ml_probability'] if c in filtered.columns]
        st.dataframe(filtered[display_cols_filtered], hide_index=True)
        st.markdown("___")

        col5, col6 = st.columns(2)

        with col5:
            st.subheader("📤 Export At-Risk Students")
            red_students = df[df['risk_flag'] == '🔴'] if 'risk_flag' in df.columns else pd.DataFrame()
            csv_bytes = red_students.to_csv(index=False).encode('utf-8')
            st.download_button("Download Red Students CSV", csv_bytes, "red_students.csv", mime="text/csv")

            st.write("-----")
            st.subheader("✉ Send Email Alert❗")
            student_to_email = st.selectbox("Select student to email", df['student_id'].tolist())
            email_input = st.text_input("Enter Guardian/Mentor Email")
            if st.button("Send Email for Selected Student"):
                if not email_input:
                    st.error("Please enter an email address.")
                else:
                    stu_row = df[df['student_id'] == student_to_email].iloc[0].to_dict()
                    # map keys expected by send_email_report
                    student_data = {
                        'student_name': stu_row.get('student_name','Unknown'),
                        'risk_score': stu_row.get('risk_score', 0),
                        'risk_flag': stu_row.get('risk_flag', 'N/A'),
                        'att_factor': stu_row.get('att_factor', 0),
                        'ass_factor': stu_row.get('ass_factor', 0),
                        'fee_factor': stu_row.get('fee_factor', 0),
                        'attempts_factor': stu_row.get('attempts_factor', 0)
                    }
                    success = send_email_report(email_input, student_data)
                    if success:
                        st.success(f"Email sent successfully to {email_input}")
                    else:
                        st.error("Failed to send email. Please check SMTP settings and st.secrets configuration.")

        with col6:
            st.subheader("📊 Student Risk Visualization")
            if isinstance(risk_count, pd.Series) and not risk_count.empty:
                color_map = {"🔴": "red", "🟡": "yellow", "🟢": "green"}
                label_map_rev = {"🔴": "High Risk", "🟡": "Medium Risk", "🟢": "Low Risk"}
                labels = [label_map_rev.get(l, l) for l in risk_count.index.tolist()]
                values = risk_count.values.tolist()
                fig = px.pie(values=values, names=labels)
                fig.update_traces(textinfo="percent+label", textfont=dict(color="black", size=14, family="Arial"))
                fig.update_layout(width=400, height=420, margin=dict(l=20, r=20, t=20, b=20), showlegend=True, legend_title="Risk Levels")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No risk-flag data available to plot.")

    else:
        st.info("Please upload all required files to see the dashboard.")

# ---------------------------
# Dashboard (aggregate view)
# ---------------------------
elif option == "Dashboard":
    st.subheader("📊 Student Dashboard")

    att = st.session_state.get('att_df')
    ass = st.session_state.get('ass_df')
    fees = st.session_state.get('fees_df')

    if att is not None and ass is not None and fees is not None:
        try:
            model_results = predict_risk()
            if not (isinstance(model_results, pd.DataFrame) and 'final_dropout_probability' in model_results.columns):
                model_results = None
        except Exception:
            model_results = None

        df = prepare_risk_dataframe(att, ass, fees, model_results, config)

        total_stu = len(df)
        risk_count = df['risk_flag'].value_counts()
        high = risk_count.get("🔴", 0)
        medium = risk_count.get("🟡", 0)
        low = risk_count.get("🟢", 0)

        col1, col2 = st.columns(2)
        with col1:
            col3, col4 = st.columns(2)
            with col3:
                st.metric("👥 Total Students", total_stu)
                st.metric("🟡 Medium Risk Students", medium)
            with col4:
                st.metric("🔴 High Risk Students", high)
                st.metric("🟢 Low Risk Students", low)
        with col2:
            pct = (high / total_stu * 100) if total_stu > 0 else 0
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=pct, title={'text': "High Risk Indicator"},
                delta={'reference': 20}, number={'suffix': "%"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "red"},
                       'steps': [
                           {'range': [0, 33], 'color': "lightgreen"},
                           {'range': [33, 66], 'color': "yellow"},
                           {'range': [66, 100], 'color': "red"}]}
            ))
            fig.update_layout(height=240, margin=dict(l=5, r=5, t=40, b=5), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        st.write("_")
        st.subheader("Student Detail Analysis")
        student_choice = st.selectbox("*Select Student*", df['student_id'].unique())
        stu_data = df[df['student_id'] == student_choice].iloc[0]
        st.write(f"### {stu_data['student_name']}")
        st.write(f"Risk Score: {stu_data['risk_score']:.2f} ({stu_data['risk_flag']})")
        st.write(f"Attendance %: {stu_data.get('attendance_percent', 0):.2f}")
        st.write(f"Avg Score: {stu_data.get('avg_score', 0):.2f}")
        st.write("_")
    else:
        st.info("Please upload all required files to see the dashboard.")
