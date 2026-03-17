import streamlit as st
import os
import re
import tempfile
import smtplib
import datetime
import uuid
from email.message import EmailMessage

import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AI Recruitment Dashboard",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Recruitment Screening Dashboard")
st.caption("Automated screening with interview scheduling & AI insights")
st.divider()


# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# -------------------- HELPERS --------------------
def extract_text(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def extract_email(text):
    match = re.search(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        text
    )
    return match.group(0) if match else None


def generate_interview_link():
    unique_id = str(uuid.uuid4())[:8]
    link = f"https://teams.microsoft.com/l/meetup-join/{unique_id}"
    expiry = datetime.datetime.now() + datetime.timedelta(days=1)
    return link, expiry.strftime("%Y-%m-%d %H:%M")


def send_status_email(to_email, status, candidate_name, interview_time=None):
    sender_email = st.secrets["EMAIL"]
    sender_password = st.secrets["EMAIL_PASSWORD"]
    interviewer_email = st.secrets["INTERVIEWER_EMAIL"]

    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = to_email

    if status == "Selected":
        interview_link, expiry_time = generate_interview_link()

        msg["Subject"] = "Interview Invitation – AI Recruitment"
        msg.set_content(
            f"Dear Candidate,\n\n"
            f"Congratulations! You have been shortlisted.\n\n"
            f"Interview Time: {interview_time}\n\n"
            f"Join your interview:\n{interview_link}\n\n"
            f"This link is valid until: {expiry_time}\n\n"
            f"Regards,\nHR Team"
        )

        # -------- INTERVIEWER ALERT --------
        alert = EmailMessage()
        alert["From"] = sender_email
        alert["To"] = interviewer_email
        alert["Subject"] = "New Interview Scheduled"

        alert.set_content(
            f"Candidate: {candidate_name}\n"
            f"Interview Time: {interview_time}\n"
            f"Meeting Link: {interview_link}"
        )

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.send_message(alert)

    else:
        msg["Subject"] = "Application Status Update"
        msg.set_content(
            "Dear Candidate,\n\n"
            "Thank you for your interest. We will not proceed further.\n\n"
            "Regards,\nHR Team"
        )

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)


def generate_ai_explanation(jd_text, resume_text, percentage, status):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    prompt = f"""
You are an AI recruitment assistant.

Job Description:
{jd_text}

Candidate Resume:
{resume_text}

Match Score: {percentage}%
Final Decision: {status}

Give 3 short bullet points explaining the decision.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content


# -------------------- INPUT UI --------------------
col1, col2 = st.columns([2, 1])

with col1:
    job_description = st.text_area(
        "📌 Job Description",
        height=220,
        placeholder="Paste job description here..."
    )

with col2:
    threshold = st.slider("🎯 Threshold (%)", 0, 100, 70)
    interview_time = st.datetime_input("🗓️ Select Interview Time")

st.subheader("📂 Upload Resumes")

uploaded_files = st.file_uploader(
    "Upload PDF resumes",
    type=["pdf"],
    accept_multiple_files=True
)

run = st.button("🚀 Run Screening", use_container_width=True)


# -------------------- MAIN LOGIC --------------------
accepted = []
rejected = []

if run and job_description and uploaded_files:

    jd_embedding = model.encode(job_description)

    for uploaded_file in uploaded_files:

        suffix = os.path.splitext(uploaded_file.name)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(uploaded_file.getvalue())
            temp_path = temp.name

        resume_text = extract_text(temp_path)

        if not resume_text.strip():
            st.warning(f"No readable text in {uploaded_file.name}")
            continue

        resume_embedding = model.encode(resume_text)

        score = cosine_similarity(
            [jd_embedding],
            [resume_embedding]
        )[0][0]

        percentage = round(score * 100, 2)

        if percentage >= threshold:
            status = "Selected"
            accepted.append((uploaded_file.name, percentage))
        else:
            status = "Rejected"
            rejected.append((uploaded_file.name, percentage))

        # -------- AI EXPLANATION --------
        explanation = generate_ai_explanation(
            job_description,
            resume_text,
            percentage,
            status
        )

        st.markdown(f"### 🧠 AI Explanation – {uploaded_file.name}")
        st.info(explanation)

        # -------- EMAIL --------
        email = extract_email(resume_text)

        if email:
            try:
                send_status_email(
                    email,
                    status,
                    uploaded_file.name,
                    interview_time
                )
                st.success(f"Email sent to {email} ({status})")
            except Exception as e:
                st.error(f"Email failed: {e}")
        else:
            st.warning("No email found in resume")


    # -------------------- DASHBOARD --------------------
    st.divider()
    st.subheader("📊 Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total", len(accepted) + len(rejected))
    c2.metric("Selected", len(accepted))
    c3.metric("Rejected", len(rejected))

    st.divider()

    # -------------------- RESULTS --------------------
    left, right = st.columns(2)

    with left:
        st.markdown("### ✅ Selected Candidates")
        if accepted:
            for name, score in accepted:
                st.success(f"{name} — {score}%")
        else:
            st.info("No selected candidates")

    with right:
        st.markdown("### ❌ Rejected Candidates")
        if rejected:
            for name, score in rejected:
                st.error(f"{name} — {score}%")
        else:
            st.info("No rejected candidates")