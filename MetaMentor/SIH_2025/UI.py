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
flask_process = subprocess.Popen([sys.executable, "app.py"])
time.sleep(2)  # give Flask a moment to start

def cleanup():
    print("🛠 Cleaning up Flask process...")
    flask_process.terminate()  # stop Flask
    flask_process.wait()
    shutil.rmtree("uploads", ignore_errors=True)
    print("✅ Cleanup done.")

atexit.register(cleanup)


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
max_attempts = st.sidebar.number_input("Max Exam Attempts Allowed", 1, 7, 4)

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
        send_to_backend(student_file)

    att_file = st.file_uploader("2. Upload Attendance Data CSV", type=["csv"], key="att_file")
    if att_file:
        st.session_state.att_df = pd.read_csv(att_file)
        send_to_backend(att_file)

    fees_file = st.file_uploader("3. Upload Fees Data CSV", type=["csv"], key="fees_file")
    if fees_file:
        st.session_state.fees_df = pd.read_csv(fees_file)
        send_to_backend(fees_file)

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
                send_to_backend(file)
            elif subject_key in st.session_state.assessment_subjects:
                pass


  

    # Info Modal
    if "notice_open" not in st.session_state:
        st.session_state.notice_open=True
        modal.open()
    if modal.is_open():
        with modal.container():
            st.markdown("""
            Ensure CSV files have correct headers!

            """)
            st.image("images\popup.jpg", 
            caption="Example CSV Format", 
            use_container_width=True)
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

    if True:
    # if required_files_present:
        st.header("🎓 Dropout Risk Prediction Table")
        st.write("⚙ Running model/heuristic...")
        try:
            res = requests.post("http://127.0.0.1:5000/save_config", json=config)
            dataf = ml_model()
            if isinstance(dataf, pd.DataFrame) and 'Risk_Level' in dataf.columns:
                st.success("✅ Model processed successfully! Using model output.")
                results_to_show = dataf.copy()
                results_to_show['Risk_Level'] = results_to_show['Risk_Level'].apply(extract_risk_emoji).astype(str)
            else:
                raise ValueError("Model not returning valid output.")
        except Exception as e:
            st.warning(f"❌ ML Model failed or unavailable. Error: {e}")

        # results_to_show['Risk_Level'] = results_to_show['Risk_Level'].astype(str)

        # st.subheader("📃 All Students Risk Data")

        # display_cols_list = ['Roll_no','Name','Risk_Level','Risk_Score','Dropout_Probability_Percentage','attendance_percentage','fee_paid_ratio','avg_all_marks','total_failed_attempts','subject_with_failures','failure_rate', 'Recommendations']
        # display_cols = [c for c in display_cols_list if c in results_to_show.columns]
        
        # display_df = results_to_show[display_cols].copy()
        # if 'Risk_Score' in display_df.columns:
        #     display_df['Risk_Score'] = display_df['Risk_Score'].map('{:.2f}'.format)
        # if 'Dropout_Probability_Percentage' in display_df.columns:
        #     display_df['Dropout_Probability_Percentage'] = display_df['Dropout_Probability_Percentage'].map('{:.1f}%'.format)
        # if 'attendance_percentage' in display_df.columns:
        #     display_df['attendance_percentage'] = (display_df['attendance_percentage'].astype(float) * 100).map('{:.1f}%'.format)
        # if 'fee_paid_ratio' in display_df.columns:
        #     display_df['fee_paid_ratio'] = display_df['fee_paid_ratio'].map('{:.2f}'.format)
        # if 'avg_all_marks' in display_df.columns:
        #     display_df['avg_all_marks'] = display_df['avg_all_marks'].map('{:.1f}'.format)

        # # display_df = pd.read_csv('student_dropout_risk_analysis.csv')
        st.dataframe(results_to_show, hide_index=True)
        # st.markdown("___")


        
        # # Filters
        # st.subheader("👥 Filter Students")
        # risk_count = results_to_show['Risk_Level'].value_counts()
        # label_map = {"Low Risk": "🟢","Medium Risk": "🟡","High Risk": "🔴"}
        # risk_level_choice = st.selectbox("Select Risk Level", ["All"] + list(label_map.keys()))
        
        # if risk_level_choice != "All":
        #     emoji_value = label_map[risk_level_choice]
        #     filtered = results_to_show[results_to_show['risk_level'] == emoji_value]
        # else:
        #     filtered = results_to_show
        
        # if not filtered.empty:
        #     filtered_display_df = display_df[display_df['Roll_no'].isin(filtered['Roll_no'])].copy()
        #     st.dataframe(filtered_display_df, hide_index=True)
        # else:
        #     st.info("No students match the selected filter.")
        # st.markdown("___")