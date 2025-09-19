import os
import time
import logging
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import pandas as pd
from dotenv import load_dotenv
from typing import Dict

load_dotenv()

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("email_worker.log"), logging.StreamHandler()]
)

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
FROM_NAME = os.getenv("FROM_NAME", "Dropout Prediction System")
FROM_EMAIL = os.getenv("FROM_EMAIL", SMTP_USER)

# adjust these paths as needed (or accept arguments)
STUDENTS_CSV = "red_students.csv"   # or point to your stu_df merged export
FULL_STU_CSV = "students.csv"       # optional full merge CSV

# utility email builder
def build_message(to_email: str, student: Dict) -> MIMEMultipart:
    msg = MIMEMultipart()
    msg["From"] = f"{FROM_NAME} <{FROM_EMAIL}>"
    msg["To"] = to_email
    msg["Subject"] = f"Dropout Risk Alert — {student.get('name', 'Student')}"
    body = f"""
Dear Guardian / Mentor,

Student: {student.get('name')} ({student.get('class', '')})
Risk Score: {student.get('risk_score', 0):.2f}
Risk Level: {student.get('risk_flag')}

Key Factors:
- Attendance Factor: {student.get('att_factor', 0):.2f}
- Assessment Factor: {student.get('ass_factor', 0):.2f}
- Fee Factor: {student.get('fee_factor', 0):.2f}
- Attempts Factor: {student.get('attempts_factor', 0):.2f}

Please consider early intervention.

Regards,
{FROM_NAME}
"""
    msg.attach(MIMEText(body, "plain"))
    return msg

# robust send with retries
def send_message(smtp_conn, to_email, msg, max_retries=2):
    for attempt in range(1, max_retries + 1):
        try:
            smtp_conn.send_message(msg)
            return True
        except Exception as e:
            logging.warning(f"Attempt {attempt} failed sending to {to_email}: {e}")
            time.sleep(2 ** attempt)  # exponential backoff
    return False

def load_red_students(path=STUDENTS_CSV):
    # Prefer file that already contains risk columns; otherwise you can compute risk
    if not os.path.exists(path):
        logging.error(f"File not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df

def send_bulk_emails(df: pd.DataFrame, guardian_email_col="guardian_email"):
    if df.empty:
        logging.info("No students to process.")
        return

    # Filter high-risk (assuming risk_flag column holds emoji)
    red = df[df["risk_flag"] == "🔴"].copy()
    logging.info(f"Found {len(red)} high-risk students")

    if red.empty:
        return

    # Connect SMTP once
    try:
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
    except Exception as e:
        logging.exception("Failed to connect/login to SMTP server")
        return

    sent = 0
    failed = 0
    for _, row in red.iterrows():
        to_email = row.get(guardian_email_col) or row.get("guardian") or row.get("email")
        if not to_email or pd.isna(to_email) or "@" not in str(to_email):
            logging.warning(f"Missing/invalid guardian email for student_id {row.get('student_id')}")
            failed += 1
            continue

        student = {
            "name": row.get("name"),
            "class": row.get("class"),
            "risk_score": row.get("risk_score", 0),
            "risk_flag": row.get("risk_flag"),
            "att_factor": row.get("att_factor", 0),
            "ass_factor": row.get("ass_factor", 0),
            "fee_factor": row.get("fee_factor", 0),
            "attempts_factor": row.get("attempts_factor", 0)
        }

        msg = build_message(to_email, student)
        msg["To"] = to_email
        try:
            success = send_message(server, to_email, msg)
            if success:
                logging.info(f"Email sent to {to_email} for student {student['name']}")
                sent += 1
            else:
                logging.error(f"Failed to send email to {to_email} after retries")
                failed += 1
        except Exception as e:
            logging.exception(f"Unexpected error sending to {to_email}")
            failed += 1
        time.sleep(1)  # small throttle to avoid spam limits

    server.quit()
    logging.info(f"Done. Sent={sent}, Failed={failed}")

if __name__ == "__main__":
    # simple run — you can wrap this with scheduling
    df = load_red_students(STUDENTS_CSV)
    send_bulk_emails(df)
