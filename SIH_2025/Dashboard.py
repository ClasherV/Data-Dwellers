import streamlit as st #type:ignore
import numpy as np #type:ignore
import pandas as pd #type:ignore
from datetime import datetime
import plotly.express as px #type:ignore
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import plotly.graph_objects as go

import requests

def send_to_backend(uploaded_file):
    if uploaded_file is not None:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        response = requests.post("http://127.0.0.1:5000/upload", files=files)
        if response.status_code == 200:
            st.success("✅ File saved to backend")
            return response.json()["path"]
        else:
            st.error("❌ Failed to upload to backend")
            return None


image_path="images\\BG.jpg"

with open(image_path,"rb") as f:
    encoded_img=base64.b64encode(f.read()).decode()

#bg image
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
#sidebar color
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
#003300

st.set_page_config(page_title="MetaMentor", layout="wide")
st.title("AI-based Dropout Prediction & Counseling System")


def compute_factors(row, config):
    # Attendance (attendance_percent preferred, fallback to attendance)
    att = row.get('attendance_percent', row.get('attendance', 1.0))
    try:
        att = float(att)
    except Exception:
        att = 1.0
    att_threshold = config.get('att_threshold', 0.75) or 1.0
    att_factor = max(0, (att_threshold - att) / att_threshold)

    # Assessment: use score_threshold (your sidebar) and avg_score from ass_df
    avg = row.get('avg_score', row.get('score_percent', 1.0))
    try:
        avg = float(avg)
    except Exception:
        avg = 1.0
    score_threshold = config.get('score_threshold', 0.4) or 1.0
    ass_factor = max(0, (score_threshold - avg) / score_threshold)

    # Fees: prefer fraction_unpaid if present, otherwise compute from fee_paid_ratio
    fee_paid = row.get('fees_paid_ratio', None)
    if fee_paid is None or (isinstance(fee_paid, float) and np.isnan(fee_paid)):
        frac_unpaid = row.get('fraction_unpaid', 0.0)
    else:
        try:
            frac_unpaid = 1.0 - float(fee_paid)
        except Exception:
            frac_unpaid = row.get('fraction_unpaid', 0.0)
    fee_factor = max(0, float(frac_unpaid))

    # Attempts: use attempts_used or num_failed_attempts, normalized by max_attempts
    attempts = row.get('attempts_used', row.get('num_failed_attempts', 0))
    try:
        attempts = float(attempts)
    except Exception:
        attempts = 0.0
    max_att = config.get('max_attempts', 5) or 1.0
    attempts_factor = min(attempts / max_att, 1.0)

    return att_factor, ass_factor, fee_factor, attempts_factor


def compute_risk(row, config):
    att_factor, ass_factor, fee_factor, attempts_factor = compute_factors(row, config)
    risk = (config['w_att'] * att_factor + config['w_ass'] * ass_factor +
            config['w_fee'] * fee_factor + config['w_attempts'] * attempts_factor)
    if risk < 0.4:
        flag = "🟢"
    elif risk < 0.7:
        flag = "🟡"
    else:
        flag = "🔴"
    return risk, flag, att_factor, ass_factor, fee_factor, attempts_factor

def send_email_report(to_email, student_data):
    sender_email = "youremail@example.com"  # replace with your email
    sender_password = "yourpassword"        # replace with your password or app password
    subject = f"Dropout Risk Alert: {student_data['name']} ({student_data['class']})"

    body = f"""
    Dear Mentor/Guardian,

    Student: {student_data['name']} ({student_data['class']})
    Risk Score: {student_data['risk_score']:.2f}
    Risk Level: {student_data['risk_flag']}

    Key Factors:
    - Attendance Factor: {student_data['att_factor']:.2f}
    - Assessment Factor: {student_data['ass_factor']:.2f}
    - Fee Factor: {student_data['fee_factor']:.2f}
    - Attempts Factor: {student_data['attempts_factor']:.2f}

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
        return False

#sidebar
st.sidebar.header("Configuration")
st.sidebar.markdown("**➡️ Set Threshold**")

#threshold inputs
att_threshold = st.sidebar.slider("Attendance Threshold", 0.0, 1.0, 0.75)
score_threshold = st.sidebar.slider("Score Threshold", 0.0, 1.0, 0.4)
fee_overdue_days = st.sidebar.number_input("Fee Overdue Days Threshold", 0, 180, 30)
max_attempts = st.sidebar.number_input("Max Exam Attempts Allowed", 1, 10, 5)

st.sidebar.markdown("**➡️ Set Weightage**")

#weightage inputs
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

option = st.selectbox(
    "",["Upload File","Performance Table","Dashboard"],
    index=0  # by default upload file selected
)

def highlight_risk(row):
    if row["risk_flag"] == "🔴":
        color = "background-color: #f37262; color: black;"
    elif row["risk_flag"] == "🟡":
        color = "background-color:#e7e06f; color: black;"
    elif row["risk_flag"] == "🟢":
        color = "background-color: #65d85f; color: black;"
    else:
        color = ""
    return [color] * len(row)

if option == "Upload File":
    st.subheader("📂 Upload Student Data")

    uploaded_att = st.file_uploader("Upload Attendance CSV", type=["csv"], key="att_file")
    if uploaded_att is not None:
        st.session_state['att_df'] = pd.read_csv(uploaded_att)
        backend_path = send_to_backend(uploaded_att)  # saves on Flask server
        st.write("Saved at:", backend_path)

    uploaded_ass = st.file_uploader("Upload Assessment CSV", type=["csv"], key="ass_file")
    if uploaded_ass is not None:
        st.session_state['ass_df'] = pd.read_csv(uploaded_ass)
        backend_path = send_to_backend(uploaded_ass)  # saves on Flask server
        st.write("Saved at:", backend_path)
    
    uploaded_fees = st.file_uploader("Upload Fees CSV", type=["csv"], key="fees_file")
    if uploaded_fees is not None:
        st.session_state['fees_df'] = pd.read_csv(uploaded_fees)
        backend_path = send_to_backend(uploaded_fees)  # saves on Flask server
        st.write("Saved at:", backend_path)

elif option == "Performance Table":
    att = st.session_state.get('att_df')
    ass = st.session_state.get('ass_df')
    fees = st.session_state.get('fees_df')
    # stu = st.session_state.get('stu_df')

    if att is not None and ass is not None and fees is not None:

        #avg % of numerical col
        att = att.groupby('student_id').agg({
            'attendance_percent': 'mean',   # average attendance
            'student_name': 'first'         # keep the name from the file
        }).reset_index()

        ass = ass[['student_id', 'avg_score']].copy()


        # fees['due_date'] = pd.to_datetime(fees['due_date'])
        today = datetime.today()

        fees_agg = fees[['student_id', 'fees_paid_ratio']].copy()
        fees_agg['overdue_days'] = 0  # no due_date in dataset
        fees_agg['fraction_unpaid'] = 1 - fees_agg['fees_paid_ratio']  # unpaid = inverse of paid ratio

        df = att.merge(ass, on="student_id", how="left").merge(fees_agg, on="student_id", how="left")



        #default values for missing data
        df = df.fillna({'attendance':1.0, 'avg_score':1.0, 'overdue_days':0, 'fraction_unpaid':0, 'attempts_used':0})

        if 'num_failed_attempts' in ass.columns:
            df['attempts_used'] = df['num_failed_attempts']
        else:
            df['attempts_used'] = 0

        
        risk_results = df.apply(lambda r: compute_risk(r, config), axis=1)
        df[['risk_score','risk_flag','att_factor','ass_factor','fee_factor','attempts_factor']] = pd.DataFrame(risk_results.tolist(), index=df.index)


        st.subheader("📃 All Students Risk Data")
        if 'ml_probability' in df.columns:
            st.dataframe(df[['student_id','student_name','attendance_percent','avg_score','risk_score','risk_flag']], hide_index=True)
        else:
            st.dataframe(df[['student_id','student_name','attendance_percent','avg_score','risk_score','risk_flag']], hide_index=True)
        st.markdown("**_____________________________________________________________________________________________________________________________________**")


        # Filters
        st.subheader("👥 Filter Students")
        risk_count=df["risk_flag"].value_counts()
        label_map = {"Low Risk": "🟢","Medium Risk": "🟡","High Risk": "🔴"}
        risk_level = st.selectbox("Select Risk Level", ["All"] + list(label_map.keys()))
        if risk_level != "All":
            emoji_value = label_map[risk_level]
            filtered = df[df['risk_flag'] == emoji_value]
        else:
            filtered = df
        if 'ml_probability' in filtered.columns:
            st.dataframe(df[['student_id','student_name','attendance_percent','avg_score','risk_score','risk_flag']], hide_index=True)
        else:
            st.dataframe(df[['student_id','student_name','attendance_percent','avg_score','risk_score','risk_flag']], hide_index=True)

        st.markdown("**_____________________________________________________________________________________________________________________________________**")

        col5, col6=st.columns(2)
        with col5:
            # Export
            st.subheader("📤 Export At-Risk Students")
            red_students = df[df['risk_flag'] == '🔴']
            st.download_button("Download Red Students CSV", red_students.to_csv(index=False), "red_students.csv")

            st.write("-----")
            # Email Notification
            st.subheader("✉️ Send Email Alert❗")
            email_input = st.text_input("Enter Guardian/Mentor Email")
            if st.button("Send Email for Selected Student"):
                success = send_email_report(email_input, stu_data)
                if success:
                    st.success(f"Email sent successfully to {email_input}")
                else:
                    st.error("Failed to send email. Please check SMTP settings.")

        with col6:
            st.subheader("📊 Student Risk Visualization")
            st.write("")

            color_map = {"🔴": "red","🟡": "yellow","🟢": "green"}

            label_map = {"🔴": "High Risk","🟡": "Medium Risk","🟢": "Low Risk"}

            # converting index to list for mapping
            labels = risk_count.index.tolist()
            values = risk_count.values.tolist()

            display_labels = [label_map[label] for label in labels]

            # pie chart
            fig = px.pie(names=display_labels,values=values,color=labels,color_discrete_map=color_map)

            fig.update_traces(
                textinfo="percent+label",
                textfont=dict(color="black", size=14, family="Arial", weight="bold")
            )

            fig.update_layout(width=310,height=310,margin=dict(l=10, r=10, t=10, b=10),
            showlegend=True,legend_title="Risk Levels")

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Please upload all required files to see the dashboard.")
    

elif option == "Dashboard":
    st.subheader("📊 Student Dashboard")

    att = st.session_state.get('att_df')
    ass = st.session_state.get('ass_df')
    fees = st.session_state.get('fees_df')
    stu = st.session_state.get('stu_df')

    if att is not None and ass is not None and fees is not None:

        #avg % of numerical col
        att = att.groupby('student_id').agg({
            'attendance_percent': 'mean',   # average attendance
            'student_name': 'first'         # keep the name from the file
        }).reset_index()


        ass = ass[['student_id', 'avg_score']].copy()


        # fees['due_date'] = pd.to_datetime(fees['due_date'])
        today = datetime.today()

        fees_agg = fees[['student_id', 'fees_paid_ratio']].copy()
        fees_agg['overdue_days'] = 0  # no due_date in dataset
        fees_agg['fraction_unpaid'] = 1 - fees_agg['fees_paid_ratio']  # unpaid = inverse of paid ratio


        #“CSV must contain columns: student_id, unpaid_amount, paid_amount…”

        #merging into stu csv
        df = att.merge(ass, on='student_id', how='left').merge(fees, on='student_id', how='left')
        df = df.merge(ass, on='student_id', how='left')
        df = df.merge(fees_agg, on='student_id', how='left')

        #default values for missing data
        df = df.fillna({'attendance':1.0, 'avg_score':1.0, 'overdue_days':0, 'fraction_unpaid':0, 'attempts_used':0})

        if 'num_failed_attempts' in ass.columns:
            df['attempts_used'] = df['num_failed_attempts']
        else:
            df['attempts_used'] = 0


        risk_results = df.apply(lambda r: compute_risk(r, config), axis=1)
        df[['risk_score','risk_flag','att_factor','ass_factor','fee_factor','attempts_factor']] = pd.DataFrame(risk_results.tolist(), index=df.index)
         
        st.write("___________")
        #kpis
        total_stu=len(df)
        risk_count=df["risk_flag"].value_counts()
        high=risk_count.get("🔴",0)
        medium=risk_count.get("🟡",0)
        low=risk_count.get("🟢",0)
        col1,col2=st.columns(2)
        with col1:
            col3,col4=st.columns(2)
            with col3:
                st.metric("**👥 Total Students**", total_stu)
                st.metric("**🟡 Medium Risk Students**", medium)
            with col4:
                st.metric("**🔴 High Risk Students**", high)
                st.metric("**🟢 Low Risk Students**", low)
        with col2:
            fig=go.Figure(go.Indicator(
                mode="gauge+number+delta", value=high/total_stu*100, title={'text':"High Risk Indicator"},
                delta={'reference':20}, number={'suffix': "%"},
                gauge={'axis':{'range':[0,100]},
                        'bar': {'color': "red"},
                        'steps': [
                            {'range': [0, 33], 'color': "lightgreen"},
                            {'range': [33, 66], 'color': "yellow"},
                            {'range': [66, 100], 'color': "red"}]}
                            ))
            fig.update_layout(width=150, height=150, margin=dict(l=5, r=5, t=40, b=5), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        st.write("___________")

        #visualization
        st.subheader("Student Detail Analysis")
        student_choice = st.selectbox("**Select Student**", df['student_id'].unique())
        stu_data = df[df['student_id'] == student_choice].iloc[0]
        st.write(f"### {stu_data['student_name']}")
        st.write(f"Risk Score: {stu_data['risk_score']:.2f} ({stu_data['risk_flag']})")
        if 'ml_probability' in stu_data:
            st.write(f"ML Predicted Dropout Probability: {stu_data['ml_probability']:.2f}")
        st.write("___________")
    else:
        st.info("Please upload all required files to see the dashboard.")


import streamlit as st
import pandas as pd
from Model import predict_risk   # import your function

st.title("🎓 Dropout Risk Prediction Dashboard")

uploaded_attendance = st.file_uploader("Upload Attendance CSV", type="csv")
uploaded_fees = st.file_uploader("Upload Fees CSV", type="csv")
uploaded_scores = st.file_uploader("Upload Assessment CSV", type="csv")

if uploaded_attendance and uploaded_fees and uploaded_scores:
    # Save uploaded files into the "uploads" folder
    with open("uploads/attendance.csv", "wb") as f:
        f.write(uploaded_attendance.read())
    with open("uploads/fees.csv", "wb") as f:
        f.write(uploaded_fees.read())
    with open("uploads/assessment.csv", "wb") as f:
        f.write(uploaded_scores.read())

    # Run model
    st.write("⚙️ Running model...")
    results = predict_risk()   # returns the DataFrame

    # Show results in frontend
    st.success("✅ Model processed successfully! Here are the at-risk students:")
    st.dataframe(results)

    # Option to download as Excel/CSV
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download At-Risk Students (CSV)",
        data=csv,
        file_name="at_risk_students.csv",
        mime="text/csv",
    )
