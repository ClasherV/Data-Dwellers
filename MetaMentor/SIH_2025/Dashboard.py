import streamlit as st #type:ignore
import numpy as np #type:ignore
import pandas as pd #type:ignore
from datetime import datetime
import plotly.express as px #type:ignore
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import plotly.graph_objects as go #type:ignore
import requests #type:ignore
from streamlit_modal import Modal #type:ignore
from TestingModel import ml_model
import re
import subprocess
import sys
import time
import atexit
import shutil

# Start Flask backend
# flask_process = subprocess.Popen([sys.executable, "app.py"])
# time.sleep(2)  # give Flask a moment to start

# def cleanup():
#     print("🛠 Cleaning up Flask process...")
#     flask_process.terminate()  # stop Flask
#     flask_process.wait()
#     shutil.rmtree("uploads", ignore_errors=True)
#     print("✅ Cleanup done.")

# atexit.register(cleanup)


#themes
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True
if 'assessment_subjects' not in st.session_state:
    st.session_state.assessment_subjects = {}

base_css = """
    <style>
        .stApp {
            font-size: 1.1rem; /* Increase base font size */
            line-height: 1.6; /* Increase spacing between lines */
        }

        label, h1, h2, h3, h4, h5, h6 {
            margin-bottom: 0.5rem !important; /* Add spacing below labels and headers */
        }
    </style>
"""

light_mode_css = base_css + """
    <style>
        .stApp {
            background-color: #FFFFFF; /* White background */
            color: #000000; /* Black text */
        }
        [data-testid="stSidebar"] {
            background-color: #F0F2F6; /* A nice light grey for the sidebar */
            color: #000000; /* Black text in the sidebar */
        }
        
        label {
            color: #000000 !important; 
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #000000 !important; /* Black for headers */
        }

        .st-b3, .st-b4, .st-b5, .st-b6, .st-b7 {
            color: #000000;
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: #F0F0F0;
        }
        .stTabs [data-baseweb="tab-list"] button {
            color: #444444; /* Dark grey for unselected tabs */
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            color: #000000;
            border-bottom: 2px solid #000000;
        }
        
        [data-testid="stAlert"] div {
            color: #000000 !important; /* Pure black for text inside alerts */
        }
        
        [data-testid="stSidebar"] input[type="number"] {
            color: #FFFFFF !important;
        }

        [data-testid="stFileUploader"] button {
            background-color: #31333F;
            color: #FFFFFF !important;
            border: 1px solid #FFFFFF;
        }
    </style>"""
dark_mode_css = base_css + """
    <style>
        .stApp {
            background-color: #1c1c1e;
            color: #f0f0f0;
        }
        [data-testid="stSidebar"] {
            background-color: #262730; /* A slightly different dark shade for sidebar */
        }

        [data-testid="stSidebar"] label {
            color: #f0f0f0 !important;
        }

        [data-testid="stSidebar"] input[type="number"] {
            color: #FFFFFF !important;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
        }

        .st-b3, .st-b4, .st-b5, .st-b6, .st-b7 {
            color: #f0f0f0;
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: #2c2c2e;
        }
        .stTabs [data-baseweb="tab-list"] button {
            color: #a0a0a0;
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            color: #ffffff;
            border-bottom: 2px solid #ffffff;
        }
        .stSlider > div > div > div[role="slider"] {
            background-color: #555;
        }

        [data-testid="stFileUploader"] button {
            background-color: #31333F;
            color: #FFFFFF !important;
            border: 1px solid #FFFFFF;
        }
    </style>"""

if st.session_state.dark_mode:
    st.markdown(dark_mode_css, unsafe_allow_html=True)
else:
    st.markdown(light_mode_css, unsafe_allow_html=True)

modal=Modal("⚠️ Important Notice", key="notice", max_width=870)

def send_to_backend(uploaded_file):

    if uploaded_file is not None:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        try:
            response = requests.post("http://127.0.0.1:5000/upload", files=files) 
            if response.status_code == 200:
                st.success(f"✅ File {uploaded_file.name} saved to backend")
                return response.json().get("path")
            else:
                st.error(f"❌ Failed to upload {uploaded_file.name}. Status: {response.status_code}", icon='❌')
                return None
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Backend server not reachable. Skipping file upload.", icon='⚠')
            return "Local_Path_Simulated"
        except Exception:
            st.error("❌ An error occurred during backend upload.", icon='❌')
            return None


def process_attendance(att_df):
    """
    Calculates attendance percentage.
    """
    date_cols = [col for col in att_df.columns if '/' in col or '-' in col or pd.api.types.is_datetime64_any_dtype(pd.to_datetime(col, errors='ignore'))]
    
    if not date_cols and len(att_df.columns) > 2:
        date_cols = att_df.columns[2:].tolist() 

    if not date_cols:
        return pd.DataFrame({'Roll_no': att_df['Roll_no'].unique(), 'attendance_percentage': 1.0})

    attendance_data = att_df[['Roll_no', 'Name'] + date_cols].copy()
    
    # standardize attendance markers: P/1 -> 1 (Present), A/0 -> 0 (Absent)
    for col in date_cols:
        attendance_data[col] = attendance_data[col].astype(str).str.upper().replace({'P': 1, 'A': 0, '1': 1, '0': 0}).fillna(0)
    
    attendance_data['total_days'] = len(date_cols)
    attendance_data['present_days'] = attendance_data[date_cols].sum(axis=1)
    
    attendance_data['attendance_percentage'] = attendance_data.apply(
        lambda row: row['present_days'] / row['total_days'] if row['total_days'] > 0 else 1.0, axis=1
    )
    
    agg_att = attendance_data.groupby('Roll_no').agg(
        Name=('Name', 'first'),
        attendance_percentage=('attendance_percentage', 'mean')
    ).reset_index()

    return agg_att[['Roll_no', 'Name', 'attendance_percentage']]

def process_assessment_new(subject_dfs, roll_df):
    """
    Aggregates assessment scores from multiple subject files.
    """
    valid_subject_dfs = {k: df for k, df in subject_dfs.items() if df is not None and not df.empty}
    
    if not valid_subject_dfs:
        return roll_df[['Roll_no', 'Name']].copy().assign(
            avg_all_marks=0.0, total_failed_attempts=0, marks_below_40=0
        )

    all_scores = []
    
    for key, df in valid_subject_dfs.items():
        score_col = next((col for col in df.columns if 'score' in col.lower() or 'mark' in col.lower()), None)
        
        if 'Roll_no' not in df.columns or score_col is None:
            continue
            
        temp_df = df[['Roll_no', score_col]].copy()
        
        failed_attempts_col = next((col for col in df.columns if 'failed_attempts' in col.lower() or 'attempts' in col.lower()), None)
        temp_df['Failed_Attempts'] = pd.to_numeric(df.get(failed_attempts_col, 0), errors='coerce').fillna(0)
        
        temp_df['Score'] = pd.to_numeric(temp_df[score_col], errors='coerce')
        temp_df = temp_df.dropna(subset=['Score'])
        
        temp_df = temp_df.groupby('Roll_no').agg({
            'Score': 'mean',
            'Failed_Attempts': 'sum'
        }).reset_index()
        
        temp_df = temp_df.rename(columns={'Score': key + '_Score', 'Failed_Attempts': key + '_Attempts'})
        all_scores.append(temp_df[['Roll_no', key + '_Score', key + '_Attempts']])

    if not all_scores:
        return roll_df[['Roll_no', 'Name']].copy().assign(
            avg_all_marks=0.0, total_failed_attempts=0, marks_below_40=0
        )

    final_ass_df = roll_df[['Roll_no', 'Name']].copy()
    for df in all_scores:
        final_ass_df = final_ass_df.merge(df, on='Roll_no', how='left')

    score_cols = [col for col in final_ass_df.columns if '_Score' in col]
    attempt_cols = [col for col in final_ass_df.columns if '_Attempts' in col]
    
    final_ass_df[score_cols] = final_ass_df[score_cols].fillna(0)
    final_ass_df[attempt_cols] = final_ass_df[attempt_cols].fillna(0)

    def calculate_mean_safe(row):
        scores = [row[col] for col in score_cols if row[col] > 0]
        return np.mean(scores) if scores else 0.0

    final_ass_df['avg_all_marks'] = final_ass_df.apply(calculate_mean_safe, axis=1)

    final_ass_df['total_failed_attempts'] = final_ass_df[attempt_cols].sum(axis=1)
    
    final_ass_df['marks_below_40'] = final_ass_df[score_cols].apply(lambda row: (row < 40).sum(), axis=1)

    return final_ass_df[['Roll_no', 'Name', 'avg_all_marks', 'total_failed_attempts', 'marks_below_40']]


def process_fees(fees_df):
    """Calculates fee paid ratio and payment delay days."""
    
    col_map = {
        'total fee': 'Total Fee',
        'paid fee': 'Paid Fee',
        'last date': 'Last Date',
        'payment date': 'Payment Date'
    }
    
    for lower_col, standard_col in col_map.items():
        if standard_col not in fees_df.columns:
            matched_col = next((c for c in fees_df.columns if lower_col in c.lower()), None)
            if matched_col:
                fees_df = fees_df.rename(columns={matched_col: standard_col})
            else:
                 fees_df[standard_col] = 0 if 'Fee' in standard_col else (datetime.now().strftime('%Y-%m-%d') if 'Date' in standard_col else "")

    required_cols = ['Roll_no', 'Name', 'Total Fee', 'Paid Fee', 'Last Date', 'Payment Date']
    
    fees_data = fees_df[[c for c in required_cols if c in fees_df.columns]].copy()
    
    fees_data['Total Fee'] = pd.to_numeric(fees_data.get('Total Fee', 0), errors='coerce').fillna(0)
    fees_data['Paid Fee'] = pd.to_numeric(fees_data.get('Paid Fee', 0), errors='coerce').fillna(0)
    
    fees_data = fees_data.groupby('Roll_no').agg({
        'Name': 'first',
        'Total Fee': 'sum',
        'Paid Fee': 'sum',
        'Last Date': 'max',
        'Payment Date': 'max'
    }).reset_index()

    fees_data['fee_paid_ratio'] = fees_data.apply(
        lambda row: row['Paid Fee'] / row['Total Fee'] if row['Total Fee'] > 0 else 1.0, axis=1
    )
    
    fees_data['Last Date'] = pd.to_datetime(fees_data['Last Date'], errors='coerce')
    fees_data['Payment Date'] = pd.to_datetime(fees_data['Payment Date'], errors='coerce')
    
    now_date = datetime.now().date() 
    
    def calculate_delay(row):
        last_date = row['Last Date'].date() if pd.notna(row['Last Date']) else None
        payment_date = row['Payment Date'].date() if pd.notna(row['Payment Date']) else None
        
        if payment_date and last_date and (payment_date > last_date):
            return (payment_date - last_date).days
        elif last_date and (row['Paid Fee'] < row['Total Fee']):
            delay = (now_date - last_date).days
            return max(0, delay)
        return 0

    fees_data['payment_delay_days'] = fees_data.apply(calculate_delay, axis=1)
    
    return fees_data[['Roll_no', 'Name', 'fee_paid_ratio', 'payment_delay_days']]

def merge_dataframes(student_df, att_df, subject_dfs, fees_df):
    """Merges all processed data into a single master DataFrame."""
    
    details_cols = ['Roll_no', 'Name', 'Email(Parent)', 'Email(Student)']

    grouping_keywords = ['class', 'course', 'year', 'batch', 'section', 'stream', 'department', 'grade']
    for col in student_df.columns:
        if col.lower() in grouping_keywords and col not in details_cols:
            details_cols.append(col)
            
    for col in ['Roll_no', 'Name']:
        if col not in student_df.columns:
            st.warning(f"Mandatory column '{col}' missing in Student Master Data. (Master Data must have Roll_no and Name)")
            return pd.DataFrame() 

    master_df = student_df[[c for c in details_cols if c in student_df.columns]].copy()
    
    att_processed = process_attendance(att_df)
    master_df = master_df.merge(att_processed[['Roll_no', 'attendance_percentage']], on='Roll_no', how='left')

    ass_processed = process_assessment_new(subject_dfs, master_df[['Roll_no', 'Name']])
    master_df = master_df.merge(ass_processed[['Roll_no', 'avg_all_marks', 'total_failed_attempts', 'marks_below_40']], on='Roll_no', how='left')

    fees_processed = process_fees(fees_df)
    master_df = master_df.merge(fees_processed[['Roll_no', 'fee_paid_ratio', 'payment_delay_days']], on='Roll_no', how='left')

    master_df = master_df.fillna({
        'attendance_percentage': 1.0,
        'avg_all_marks': 0.0,
        'fee_paid_ratio': 1.0,
        'total_failed_attempts': 0,
        'marks_below_40': 0,
        'payment_delay_days': 0
    })
    
    return master_df


def compute_factors_new(row, config):
    att = row.get('attendance_percentage', 1.0)
    att_threshold = config.get('att_threshold', 0.75)
    att_factor = max(0.0, (att_threshold - att) / (att_threshold if att_threshold != 0 else 1.0))
    
    avg = row.get('avg_all_marks', 0.0) / 100.0
    score_threshold = config.get('score_threshold', 0.4)
    ass_factor = max(0.0, (score_threshold - avg) / (score_threshold if score_threshold != 0 else 1.0))
    
    fee_paid = row.get('fee_paid_ratio', 1.0)
    frac_unpaid = 1.0 - fee_paid
    delay_days = row.get('payment_delay_days', 0)
    fee_overdue_days = config.get('fee_overdue_days', 30)
    normalized_delay = min(delay_days / fee_overdue_days, 1.0)
    fee_factor = max(frac_unpaid, normalized_delay)
    
    attempts = row.get('total_failed_attempts', 0)
    marks_below_40 = row.get('marks_below_40', 0)
    max_att = config.get('max_attempts', 5)
    attempts_factor = min(1.0, (attempts / float(max_att)) + (marks_below_40 * 0.1))
    
    return att_factor, ass_factor, fee_factor, attempts_factor

def compute_risk_new(row, config):
    att_factor, ass_factor, fee_factor, attempts_factor = compute_factors_new(row, config)
    
    w_att = config.get('w_att', 0.0)
    w_ass = config.get('w_ass', 0.0)
    w_fee = config.get('w_fee', 0.0)
    w_attempts = config.get('w_attempts', 0.0)
    
    total_weight = w_att + w_ass + w_fee + w_attempts
    
    if total_weight == 0:
        risk = 0.0
    else:
        risk = (w_att * att_factor +
                w_ass * ass_factor +
                w_fee * fee_factor +
                w_attempts * attempts_factor) / total_weight

    if risk < 0.4:
        flag = "🟢"
        recommendation = "Low Risk. Continue monitoring."
    elif risk < 0.7:
        flag = "🟡"
        recommendation = "Medium Risk. Mentor check-in needed for specific factor."
    else:
        flag = "🔴"
        recommendation = "High Risk! Immediate intervention and counseling required."

    return risk, flag, att_factor, ass_factor, fee_factor, attempts_factor, recommendation

def send_email_report(to_email, student_data):
    sender_email = st.secrets["gmail"]["sender_email"]
    sender_password = st.secrets["gmail"]["app_password"]
    
    student_name = student_data.get('Name', 'Unknown')
    risk_score = student_data.get('Risk_Score', 0)
    risk_level = student_data.get('Risk_Level', 'N/A')
    recommendation = student_data.get('Recommendations', 'N/A')
    
    att_percent = student_data.get('attendance_percentage', 0)
    fee_ratio = student_data.get('fee_paid_ratio', 0)
    failed_attempts = student_data.get('total_failed_attempts', 0)
    avg_marks = student_data.get('avg_all_marks', 0)
    
    subject = f"Dropout Risk Alert: {student_name}"

    body = f"""
    Dear Mentor/Guardian,

    Student: {student_name}
    Roll No: {student_data.get('Roll_no','N/A')}
    Risk Score: {risk_score:.2f}
    Risk Level: {risk_level}

    System Recommendation:
    {recommendation}

    Key Factors:
    - Attendance Percentage: {att_percent:.2f}%
    - Fee Paid Ratio: {fee_ratio:.2f}
    - Failed Attempts (Total): {failed_attempts}
    - Average Marks: {avg_marks:.2f}

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
        if sender_email == "youremail@example.com":
             st.warning("Skipping email send: Dummy sender email detected. Please update your `sender_email` and `sender_password` for emails to work.")
             return False
             
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Email send error: {e}")
        st.warning("Ensure you have replaced youremail@example.com and yourpassword with actual credentials and configured your email for external SMTP access (e.g., using an App Password for Gmail).")
        return False
    
#streamlit UI
st.set_page_config(page_title="MetaMentor", layout="wide")
st.title("AI-based Dropout Prediction & Counseling System")

# Sidebar Configuration
st.sidebar.header("Configuration")
button_label = "Switch to Light Mode 🌟" if st.session_state.dark_mode else "Switch to Dark Mode 🌙"
if st.sidebar.button(button_label, use_container_width=True):
    st.session_state.dark_mode = not st.session_state.dark_mode
    st.rerun()

st.sidebar.markdown("**➡️ Set Threshold**")
att_threshold = st.sidebar.slider("Attendance Threshold", 0.0, 1.0, 0.75)
score_threshold = st.sidebar.slider("Score Threshold (e.g., 0.4 for 40%)", 0.0, 1.0, 0.4)
fee_overdue_days = st.sidebar.number_input("Fee Overdue Days Threshold", 0, 180, 30)
max_attempts = st.sidebar.number_input("Max Exam Attempts Allowed", 1, 7, 5)

st.sidebar.markdown("**➡️ Set Weightages**")
w_att = st.sidebar.slider("Weight: Attendance", 0.0, 1.0, 0.3)
w_ass = st.sidebar.slider("Weight: Assessment Score", 0.0, 1.0, 0.4)
w_fee = st.sidebar.slider("Weight: Fee/Delay", 0.0, 1.0, 0.2)
w_attempts = st.sidebar.slider("Weight: Attempts/Failures", 0.0, 1.0, 0.1)
w_fee_delay = st.sidebar.slider("Weight: Fee Overdue Days", 0.0, 1.0, 0.1)

config = {
    'att_threshold': att_threshold,
    'score_threshold': score_threshold,
    'fee_overdue_days': fee_overdue_days,
    'max_attempts': max_attempts,
    'w_att': w_att,
    'w_ass': w_ass,
    'w_fee': w_fee,
    'w_fee_delay': w_fee_delay,
    'w_attempts': w_attempts
}

tab1, tab2, tab3 = st.tabs(["📂 Upload", "📊 Performance Table", "📈 Dashboard"])

def extract_risk_emoji(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    s = re.sub(r'[\uFE0F\u200B-\u200D\uFEFF]', '', s)
    for e in ("🟢", "🟡", "🔴"):
        if e in s:
            return e
    low = 'low' in s.lower()
    med = 'med' in s.lower() or 'medium' in s.lower()
    high = 'high' in s.lower()
    if s in ["🟢", "🟡", "🔴"]: return s
    if high: return "🔴"
    if med: return "🟡"
    if low: return "🟢"
    return None
#upload file section
with tab1:
    st.subheader("Upload Student Data Files")
    if 'assessment_subjects' not in st.session_state:
        st.session_state.assessment_subjects = {}
    
    student_file = st.file_uploader("1. Upload Student Master Data CSV", type=["csv"], key="student_file")
    if student_file:
        st.session_state.student_df = pd.read_csv(student_file)
        # send_to_backend(student_file)

    att_file = st.file_uploader("2. Upload Attendance Data CSV", type=["csv"], key="att_file")
    if att_file:
        st.session_state.att_df = pd.read_csv(att_file)
        # send_to_backend(att_file)

    fees_file = st.file_uploader("3. Upload Fees Data CSV", type=["csv"], key="fees_file")
    if fees_file:
        st.session_state.fees_df = pd.read_csv(fees_file)
        # send_to_backend(fees_file)

    num_subjects = st.number_input(
        "4. Specify Number of Assessment Subjects", 
        min_value=1, 
        max_value=10, 
        value=3, 
        key="num_subjects_input"
    )
    
    num_subjects = int(num_subjects)

    with st.expander(f"5. Upload Assessment Scores for {num_subjects} Subjects"):
        for i in range(1, num_subjects + 1):
            subject_key = f"subject_{i}"
            uploader_label = f"Upload Subject {i} Data CSV"
            
            file = st.file_uploader(uploader_label, type=["csv"], key=subject_key)
            
            if file:
                st.session_state.assessment_subjects[subject_key] = pd.read_csv(file)
                # send_to_backend(file)
            elif subject_key in st.session_state.assessment_subjects:
                pass
    

    


    # Info Modal
    # if "notice_open" not in st.session_state:
    #     st.session_state.notice_open=True
    #     modal.open()
    # if modal.is_open():
    #     with modal.container():
    #         st.markdown("""
    #         Ensure CSV files have correct headers!

    #         """)
    #         st.image("images\popup.jpg", 
    #         caption="Example CSV Format", 
    #         use_container_width=True)
#performance table section
with tab2:
    student = st.session_state.get('student_df')
    att = st.session_state.get('att_df')
    ass_subjects = st.session_state.get('assessment_subjects', {})
    fees = st.session_state.get('fees_df')
    
    has_valid_ass_data = any(df is not None and not df.empty for df in ass_subjects.values())
    
    required_files_present = (
        student is not None and not student.empty and
        att is not None and not att.empty and
        fees is not None and not fees.empty and 
        has_valid_ass_data
    )

    results_to_show = None 
    master_df = None 

    if required_files_present:
    # if True:
        st.header("🎓 Dropout Risk Prediction Table")

        # merge DataFrames
        try:
            master_df = merge_dataframes(student, att, ass_subjects, fees)
        except Exception as e:
            st.error(f"Error merging or processing data. Check if 'Roll_no' and other core columns exist in all files. Error: {e}")
            master_df = None

        if master_df is not None and not master_df.empty:
            try:
                model_results = pd.read_csv('student_dropout_risk_analysis.csv')
                if isinstance(model_results, pd.DataFrame) and 'Risk_Level' in model_results.columns:
                    st.success("✅ Model processed successfully! Using model output.")
                    results_to_show = model_results.copy()
                    results_to_show['Risk_Level'] = results_to_show['Risk_Level'].apply(extract_risk_emoji).astype(str)
                    if 'Dropout_Probability_Percentage' not in results_to_show.columns:
                        results_to_show['Dropout_Probability_Percentage'] = (results_to_show['Risk_Score'] * 100).round(2)
                else:
                    raise ValueError("Model not returning valid output.")
            except Exception as e:
                st.warning(f"❌ ML Model failed or unavailable. Error: {e}")
                
                # Heuristic Fallback
                risk_results = master_df.apply(lambda r: compute_risk_new(r, config), axis=1)
                (master_df['Risk_Score'], master_df['Risk_Level'], master_df['att_factor'], 
                 master_df['ass_factor'], master_df['fee_factor'], master_df['attempts_factor'],
                 master_df['Recommendations']) = zip(*risk_results)
                 
                results_to_show = master_df.rename(columns={'Name': 'Name', 'Roll_no': 'Roll_no'}).copy()
                results_to_show['Dropout_Probability_Percentage'] = (results_to_show['Risk_Score'] * 100).round(2)
            
            results_to_show['Risk_Level'] = results_to_show['Risk_Level'].astype(str)


            st.subheader("📃 All Students Risk Data")
            
            display_cols_list = ['Roll_no','Name','Dropout_Probability_Percentage','Risk_Level','Risk_Score','attendance_percentage','avg_all_marks','fee_paid_ratio','total_failed_attempts', 'Recommendations']
            display_cols = [c for c in display_cols_list if c in results_to_show.columns]
            
            display_df = results_to_show[display_cols].copy()
            if 'Risk_Score' in display_df.columns:
                display_df['Risk_Score'] = display_df['Risk_Score'].map('{:.2f}'.format)
            if 'Dropout_Probability_Percentage' in display_df.columns:
                display_df['Dropout_Probability_Percentage'] = display_df['Dropout_Probability_Percentage'].map('{:.1f}%'.format)
            if 'attendance_percentage' in display_df.columns:
                display_df['attendance_percentage'] = (display_df['attendance_percentage'].astype(float)).map('{:.1f}%'.format)
            if 'fee_paid_ratio' in display_df.columns:
                display_df['fee_paid_ratio'] = display_df['fee_paid_ratio'].map('{:.2f}'.format)
            if 'avg_all_marks' in display_df.columns:
                display_df['avg_all_marks'] = display_df['avg_all_marks'].map('{:.1f}'.format)
            
            
            st.dataframe(display_df, hide_index=True)
            st.markdown("___")
            # Filters
            st.subheader("👥 Filter Students")
            risk_count = results_to_show['Risk_Level'].value_counts()
            label_map = {"Low Risk": "🟢","Medium Risk": "🟡","High Risk": "🔴"}
            risk_level_choice = st.selectbox("Select Risk Level", ["All"] + list(label_map.keys()))
            
            if risk_level_choice != "All":
                emoji_value = label_map[risk_level_choice]
                filtered = results_to_show[results_to_show['Risk_Level'] == emoji_value]
            else:
                filtered = results_to_show
            
            if not filtered.empty:
                filtered_display_df = display_df[display_df['Roll_no'].isin(filtered['Roll_no'])].copy()
                st.dataframe(filtered_display_df, hide_index=True)
            else:
                st.info("No students match the selected filter.")
            st.markdown("___")
            

            col5, col6=st.columns(2)
            with col5:
                st.subheader("📤 Export At-Risk Students")
                red_students = results_to_show[results_to_show.get('Risk_Level') == '🔴']
                st.download_button("Download High Risk Students CSV", red_students.to_csv(index=False), "high_risk_students.csv")

                st.write("-----")
                st.subheader("✉️ Send Email Alerts ❗")
                if not results_to_show.empty:
                # Filter only medium and high risk students
                    risky_students = results_to_show[results_to_show['Risk_Level'].isin(['🟡', '🔴'])]
                
                    if not risky_students.empty:
                        if st.button("📨 Send Alerts to All Medium/High Risk Parents"):
                        # if st.button("📨 Send Test Alerts (First 3 students → My Email)"):
                            sent_count = 0
                            failed_count = 0


                            for idx, (_, row) in enumerate(risky_students.iterrows()):
                                if idx >= 3:   # stop after first 3 students
                                    break
                            
                            # for _, row in risky_students.iterrows():
                            #     roll_no = row['Roll_no']
                            #     student_details = student[student['Roll_no'] == roll_no]
                
                                roll_no = row['Roll_no']
                                student_details = student[student['Roll_no'] == roll_no]
                
                                if 'Parent_Email' in student_details.columns and not student_details.empty:
                                    parent_email = student_details['Parent_Email'].iloc[0]
                
                                    if parent_email:
                                        success = send_email_report(parent_email, row.to_dict())
                                        if success:
                                            sent_count += 1
                                        else:
                                            failed_count += 1
                                    else:
                                        failed_count += 1
                                else:
                                    failed_count += 1
                
                            st.success(f"✅ Emails sent.")
                            # st.success(f"✅ Test emails sent for {sent_count} students.")
                            if failed_count > 0:
                                st.warning(f"⚠️ Failed to send {failed_count} emails (missing/invalid addresses).")
                    else:
                        st.info("No Medium/High risk students found.")
                else:
                    st.info("No students available for emailing.")   
                st.markdown("___")           


            with col6:
                st.subheader("📊 Student Risk Visualization")
                if isinstance(risk_count, pd.Series) and not risk_count.empty:
                    color_map_emoji = {"🔴": "red","🟡": "gold","🟢": "green"} 
                    label_map_rev = {"🔴": "High Risk","🟡": "Medium Risk","🟢": "Low Risk"}
                    
                    valid_risks = risk_count.index[risk_count.index.isin(label_map_rev.keys())]
                    risk_count = risk_count[valid_risks]

                    labels = risk_count.index.tolist()
                    values = risk_count.values.tolist()
                    
                    display_labels = [label_map_rev.get(label, label) for label in labels]
                    colors = [color_map_emoji.get(label, 'grey') for label in labels]

                    fig = px.pie(names=display_labels, values=values, color=display_labels, 
                                color_discrete_map={k: v for k, v in zip(display_labels, colors)})
                                
                    fig.update_traces(textinfo="percent+label", textfont=dict(color="black", size=14, family="Arial"))
                    fig.update_layout(width=310,height=300,margin=dict(l=20, r=20, t=20, b=20), showlegend=True,legend_title="Risk Levels")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No valid risk-flag data available to plot.")
        else:
             st.info("Data processed, but the resulting DataFrame is empty. Check your input files' 'Roll_no' column for mismatches.")

    else:
        st.info("Please upload the Student Master Data, Attendance, Fees, and at least one Subject Assessment file to see the performance table.")

#dashboard section
with tab3:
    st.subheader("📊 Student Dashboard")
    if 'results_to_show' not in locals() or results_to_show is None or results_to_show.empty or 'Risk_Level' not in results_to_show.columns:
        
        student = st.session_state.get('student_df')
        att = st.session_state.get('att_df')
        ass_subjects = st.session_state.get('assessment_subjects', {})
        fees = st.session_state.get('fees_df')

        has_valid_ass_data = any(df is not None and not df.empty for df in ass_subjects.values())
        
        required_files_present = (
            student is not None and not student.empty and
            att is not None and not att.empty and
            fees is not None and not fees.empty and 
            has_valid_ass_data
        )
        
        if required_files_present:
            try:
                master_df = merge_dataframes(student, att, ass_subjects, fees)
            except Exception:
                master_df = None

            if master_df is not None and not master_df.empty:
                try:
                    model_results = ml_model()
                    if isinstance(model_results, pd.DataFrame) and 'Risk_Level' in model_results.columns:
                        results_to_show = model_results.copy()
                        results_to_show['Risk_Level'] = results_to_show['Risk_Level'].apply(extract_risk_emoji).astype(str)
                        if 'Dropout_Probability_Percentage' not in results_to_show.columns:
                            results_to_show['Dropout_Probability_Percentage'] = (results_to_show['Risk_Score'] * 100).round(2)
                    else:
                        raise ValueError("Model not returning valid output")
                except Exception:
                    # Heuristic Fallback
                    risk_results = master_df.apply(lambda r: compute_risk_new(r, config), axis=1)
                    (master_df['Risk_Score'], master_df['Risk_Level'], master_df['att_factor'], 
                     master_df['ass_factor'], master_df['fee_factor'], master_df['attempts_factor'],
                     master_df['Recommendations']) = zip(*risk_results)
                    results_to_show = master_df.rename(columns={'Name': 'Name', 'Roll_no': 'Roll_no'}).copy()
                    results_to_show['Dropout_Probability_Percentage'] = (results_to_show['Risk_Score'] * 100).round(2)
            else:
                results_to_show = None
        else:
            results_to_show = None


    if results_to_show is not None and not results_to_show.empty and 'Risk_Level' in results_to_show.columns:

        # kpis
        st.write("___")
        total_stu = len(results_to_show)
        risk_count = results_to_show['Risk_Level'].value_counts()
        high = risk_count.get("🔴",0)
        medium = risk_count.get("🟡",0)
        low = risk_count.get("🟢",0)

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
            high_risk_percent = (high/total_stu*100 if total_stu>0 else 0)
            fig=go.Figure(go.Indicator(
                mode="gauge+number+delta", value=high_risk_percent, title={'text':"High Risk %"},
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
        st.write("___")

        #student detail section
        st.subheader("Student Detail Analysis")
        student_choice = st.selectbox(
            "Select Student (Roll_no)", 
            results_to_show['Roll_no'].unique(), 
            format_func=lambda x: f"{x} - {results_to_show[results_to_show['Roll_no'] == x]['Name'].iloc[0]}", 
            key='dashboard_student_choice'
        )
        stu_data = results_to_show[results_to_show['Roll_no'] == student_choice].iloc[0]
        
        _map = {"🟢": "Low Risk", "🟡": "Medium Risk", "🔴": "High Risk"}
        emoji = extract_risk_emoji(stu_data.get('Risk_Level'))
        stu_data_risk_name = _map.get(emoji, str(stu_data.get('Risk_Level')))

        st.write(f"### {stu_data['Name']} ({stu_data['Roll_no']})")
        st.markdown(f"Risk Level: {stu_data.get('Risk_Level', 'N/A')} (Score: {stu_data.get('Risk_Score', 0):.2f})")
        st.markdown(f"Recommended Action: {stu_data.get('Recommendations', 'No specific recommendation.')}")
        st.write("---")

        st.subheader("Individual Risk Factor Contribution")

        # 2x2 Chart Layout
        chart_col1, chart_col2 = st.columns(2)
        chart_col3, chart_col4 = st.columns(2)
        
        risk_colors = {"Low Risk": "#28a745", "Medium Risk": "#ffc107", "High Risk": "#dc3545"}
        with chart_col1:
            st.markdown("##### 1. Attendance Trend Over Time 📉 (Area Chart)")
            
            att_df_clean = st.session_state.get('att_df')
            
            if att_df_clean is not None and not att_df_clean.empty and 'Roll_no' in att_df_clean.columns:
                
                date_cols = [col for col in att_df_clean.columns if '/' in col or '-' in col or pd.api.types.is_datetime64_any_dtype(pd.to_datetime(col, errors='ignore'))]
                if not date_cols and len(att_df_clean.columns) > 2:
                    date_cols = att_df_clean.columns[2:].tolist()
                
                if date_cols:
                    stu_att_df_long = att_df_clean[att_df_clean['Roll_no'] == student_choice][['Roll_no'] + date_cols].copy()
                    
                    if not stu_att_df_long.empty:
                        stu_att_df_long = stu_att_df_long.melt(id_vars=['Roll_no'], var_name='Date', value_name='Status')
                        stu_att_df_long['Date'] = pd.to_datetime(stu_att_df_long['Date'], errors='coerce')
                        stu_att_df_long['Status_Num'] = stu_att_df_long['Status'].astype(str).str.upper().replace({'P': 1, 'A': 0, '1': 1, '0': 0}).fillna(0)
                        
                        stu_att_df_long = stu_att_df_long.dropna(subset=['Date'])
                        stu_att_df_trend = stu_att_df_long.sort_values('Date').copy()
                        stu_att_df_trend['Cumulative_Present'] = stu_att_df_trend['Status_Num'].cumsum()
                        stu_att_df_trend['Cumulative_Days'] = range(1, len(stu_att_df_trend) + 1)
                        stu_att_df_trend['Attendance_Percentage'] = stu_att_df_trend['Cumulative_Present'] / stu_att_df_trend['Cumulative_Days']
                        
                        fig_att = px.area(stu_att_df_trend, x='Date', y='Attendance_Percentage',
                                        title="Cumulative Attendance % Trend",
                                        color_discrete_sequence=['#17a2b8']) 
                        
                        fig_att.update_layout(
                            yaxis_title="Attendance %", 
                            height=350, 
                            margin=dict(t=50, b=5, l=5, r=5),
                            yaxis_range=[0, 1]
                        )
                        st.plotly_chart(fig_att, use_container_width=True)
                    else:
                        st.warning("Attendance trend data not found for this student.")
                else:
                    st.info("Attendance data has no proper date columns for trend analysis.")
            else:
                st.warning("Attendance raw data (`att_df`) is missing or incomplete in session state.")

        with chart_col2:
            st.markdown("##### 2. Assessment Grade Distribution / Average Score 📚")

            ass_subjects = st.session_state.get('assessment_subjects', {})
            
            stu_ass_data = []
            for subject, df in ass_subjects.items():
                if isinstance(df, pd.DataFrame) and 'Roll_no' in df.columns and student_choice in df['Roll_no'].values:
                    stu_rows = df[df['Roll_no'] == student_choice]
                    score_col = next((col for col in stu_rows.columns if 'score' in col.lower() or 'mark' in col.lower()), None)
                    
                    if score_col:
                        scores = pd.to_numeric(stu_rows[score_col], errors='coerce').dropna().tolist()
                        for score in scores:
                            stu_ass_data.append({'Subject': subject.replace('subject_', 'Sub '), 'Score': score})
                             
            ass_df = pd.DataFrame(stu_ass_data)
            
            if not ass_df.empty and len(ass_df) > 1:
                # Plot the strip plot (Box Plot)
                fig_ass = px.strip(ass_df, y="Score", x="Subject",
                                 title="Grades Spread by Subject/Assignment",
                                 color="Subject",
                                 color_discrete_sequence=px.colors.qualitative.Dark2) 
                
                fig_ass.update_traces(jitter=1)
                fig_ass.update_layout(
                    yaxis_title="Grade/Score", 
                    xaxis_title="Assessment Category",
                    height=350, 
                    margin=dict(t=50, b=5, l=5, r=5),
                    showlegend=False
                )
                st.plotly_chart(fig_ass, use_container_width=True)
            else:
                avg_mark = stu_data.get('avg_all_marks', 0.0)
                score_threshold_val = config.get('score_threshold', 0.4) * 100
                
                bar_df = pd.DataFrame({
                    'Metric': ['Student Average Mark', 'Minimum Passing Mark'],
                    'Value': [avg_mark, score_threshold_val],
                    'Color': ['Student Score', 'Threshold']
                })

                fig_ass_bar = px.bar(
                    bar_df, 
                    x='Metric', 
                    y='Value', 
                    color='Color',
                    color_discrete_map={'Student Score': '#ffb703', 'Threshold': '#e63946'},
                    title=f"Overall Average Mark: {avg_mark:.1f}"
                )
                fig_ass_bar.update_layout(
                    yaxis_title="Score", 
                    height=350, 
                    margin=dict(t=50, b=5, l=5, r=5),
                    yaxis_range=[0, 100]
                )
                st.plotly_chart(fig_ass_bar, use_container_width=True)

        with chart_col3:
            st.markdown("##### 3. Fee Payment Status 💰 (Donut Chart)")

            paid_ratio = stu_data.get('fee_paid_ratio', 1.0)
            unpaid_ratio = 1.0 - paid_ratio
            
            fee_data_for_chart = pd.DataFrame({
                'Status': ['Fee Paid', 'Fee Unpaid'],
                'Ratio': [paid_ratio, unpaid_ratio]
            })
            
            color_map = {'Fee Paid': '#28a745', 'Fee Unpaid': '#dc3545'} 
            
            fig_fee = px.pie(
                fee_data_for_chart, 
                values='Ratio', 
                names='Status', 
                color='Status',
                color_discrete_map=color_map,
                title='Fee Paid vs. Unpaid Ratio'
            )
            
            fig_fee.update_traces(
                hole=.4, 
                textinfo='percent', 
                marker=dict(line=dict(color='#000000', width=1))
            ) 
            
            fig_fee.update_layout(
                height=350, 
                margin=dict(t=50, b=5, l=5, r=5),
                showlegend=True,
                legend=dict(orientation="h", y=-0.1) 
            )
            st.plotly_chart(fig_fee, use_container_width=True)

        with chart_col4:
            st.markdown("##### 4. Overall Risk Profile 🕸 (Radar Chart)")
            att_factor, ass_factor, fee_factor, attempts_factor = compute_factors_new(stu_data, config)

            stu_data['att_factor'] = att_factor
            stu_data['ass_factor'] = ass_factor
            stu_data['fee_factor'] = fee_factor
            stu_data['attempts_factor'] = attempts_factor

            if all(factor in stu_data for factor in ['att_factor', 'ass_factor', 'fee_factor', 'attempts_factor']):
                
                categories = ['Attendance', 'Assessment', 'Fees', 'Attempts']
                values = [
                    stu_data['att_factor'],
                    stu_data['ass_factor'],
                    stu_data['fee_factor'],
                    stu_data['attempts_factor']
                ]

                values_closed = values + [values[0]]
                categories_closed = categories + [categories[0]] 

                fig_radar = go.Figure(data=[
                    go.Scatterpolar(
                        r=values_closed,
                        theta=categories_closed,
                        fill='toself',
                        name=stu_data['Name'],
                        line_color=risk_colors.get(stu_data_risk_name, 'gray'),
                        marker={'color': risk_colors.get(stu_data_risk_name, 'gray')}
                    )
                ])

                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1],
                            tickvals=[0.0, 0.5, 1.0], 
                            tickmode='array',
                            showline=False
                        ),
                        angularaxis=dict(
                            direction = "clockwise", 
                            period = len(categories)
                        )
                    ),
                    showlegend=False,
                    title_text=f"Risk Profile (0=Low, 1=High Risk)",
                    height=350,
                    margin=dict(t=50, b=5, l=5, r=5)
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            else:

                st.warning("Risk factor breakdown not available. (Heuristic model factors missing)")



        st.markdown("---")
        st.markdown("### Cohort Analysis: Student Performance Distribution")
        
        df_for_chart = results_to_show.copy()
        
        risk_map_rev = {"🟢": "Low Risk", "🟡": "Medium Risk", "🔴": "High Risk"}
        df_for_chart['Risk_Name'] = df_for_chart['Risk_Level'].map(risk_map_rev)

        category_order = ["Low Risk", "Medium Risk", "High Risk"]
        df_for_box_plot = df_for_chart[df_for_chart['avg_all_marks'] > 0].copy()

        if not df_for_box_plot.empty:
            st.markdown("##### 5. Assessment Score Distribution by Risk Level (Box Plot)")
            risk_color_map = {"Low Risk": "green", "Medium Risk": "gold", "High Risk": "red"}
            df_for_box_plot['Risk_Name'] = pd.Categorical(df_for_box_plot['Risk_Name'], categories=category_order, ordered=True)
            df_for_box_plot = df_for_box_plot.sort_values('Risk_Name')

            fig_box = px.box(
                df_for_box_plot,
                x='Risk_Name',
                y='avg_all_marks',
                color='Risk_Name',
                category_orders={"Risk_Name": category_order},
                color_discrete_map=risk_color_map,
                title="Distribution of Average Assessment Scores Across Risk Levels",
                labels={
                    'avg_all_marks': 'Average Assessment Score (0-100)',
                    'Risk_Name': 'Predicted Dropout Risk Level'
                },
                height=550
            )

            score_threshold_val = config.get('score_threshold', 0.4) * 100
            
            fig_box.add_hline(
                y=score_threshold_val, 
                line_width=1.5, 
                line_dash="dash", 
                line_color="darkblue",
                annotation_text=f"Passing Threshold ({score_threshold_val:.0f})",
                annotation_position="top right"
            )

            fig_box.update_layout(
                yaxis_range=[0, df_for_box_plot['avg_all_marks'].max() * 1.05],
                showlegend=False
            )

            st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("##### 5. Student Risk Score Comparison (Bar Chart)")

        df_for_chart['Name_Roll'] = df_for_chart['Name'] + ' (' + df_for_chart['Roll_no'].astype(str) + ')'
        
        df_sorted = df_for_chart.sort_values('Risk_Score', ascending=False)
        
        risk_color_map = {"Low Risk": "green", "Medium Risk": "gold", "High Risk": "red"}

        fig_bar = px.bar(
            df_sorted,
            x='Name_Roll',
            y='Risk_Score',
            color='Risk_Name',
            color_discrete_map=risk_color_map,
            title="Individual Student Risk Scores (0 to 1)",
            labels={
                'Name_Roll': 'Student',
                'Risk_Score': 'Dropout Risk Score (0 = Low, 1 = High)'
            },
            height=550
        )

        fig_bar.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="High Risk Threshold", annotation_position="top left")
        fig_bar.add_hline(y=0.4, line_dash="dash", line_color="gold", annotation_text="Medium Risk Threshold", annotation_position="top left")

        fig_bar.update_layout(
            xaxis={'categoryorder':'array', 'categoryarray': df_sorted['Name_Roll'].tolist()},
            xaxis_tickangle=-45,
            showlegend=True,
            margin=dict(b=100)
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.info("Please upload all required files and ensure data is properly structured to see the dashboard.")
