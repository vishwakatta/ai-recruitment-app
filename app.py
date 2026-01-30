import streamlit as st
import os
import tempfile
import re
import smtplib
from email.message import EmailMessage

import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------- CONFIG --------------------
st.set_page_config(page_title="AI Resume Screening", layout="wide")


# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# -------------------- HELPER FUNCTIONS --------------------
def extract_text(file_path):
    text = ""
    if file_path.lower().endswith(".pdf"):
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text()
    return text


def extract_email(text):
    match = re.search(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        text
    )
    return match.group() if match else None


def send_status_email(to_email, status):
    sender_email = st.secrets["EMAIL"]
    sender_password = st.secrets["EMAIL_PASSWORD"]

    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = to_email

    if status == "Selected":
        msg["Subject"] = "Application Status – Shortlisted"
        msg.set_content(
            "Dear Candidate,\n\n"
            "We are pleased to inform you that your profile has been shortlisted.\n"
            "Our HR team will contact you for the next steps.\n\n"
            "Regards,\nHR Team"
        )
    else:
        msg["Subject"] = "Application Update"
        msg.set_content(
            "Dear Candidate,\n\n"
            "Thank you for your interest. After careful review, "
            "we will not be moving forward at this stage.\n\n"
            "Regards,\nHR Team"
        )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, sender_password)
        server.send_message(msg)


# -------------------- UI --------------------
st.title("AI-Powered Resume Screening")

jd = st.text_area("Paste Job Description")

threshold = st.slider("Shortlisting Threshold (%)", 0, 100, 80)

uploaded_files = st.file_uploader(
    "Upload Resumes (PDF only)",
    type=["pdf"],
    accept_multiple_files=True
)

run = st.button("Run Screening")


# -------------------- MAIN LOGIC --------------------
if run:
    if not jd or not uploaded_files:
        st.warning("Please provide Job Description and upload resumes.")
        st.stop()

    jd_embedding = model.encode(jd)

    accepted = []
    rejected = []

    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            file_bytes = uploaded_file.getvalue()
            temp.write(file_bytes)
            temp_path = temp.name

        if os.path.getsize(temp_path) == 0:
            st.error(f"{uploaded_file.name} is empty or corrupted.")
            continue

        resume_text = extract_text(temp_path)

        if not resume_text.strip():
            st.error(f"No readable text found in {uploaded_file.name}")
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

        email = extract_email(resume_text)

        if email:
            try:
                send_status_email(email, status)
                st.success(f"Email sent to {email} ({status})")
            except Exception as e:
                st.error(f"Email failed for {email}: {e}")
        else:
            st.warning(f"No email found in {uploaded_file.name}")

    # -------------------- RESULTS --------------------
    accepted.sort(key=lambda x: x[1], reverse=True)
    rejected.sort(key=lambda x: x[1], reverse=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Accepted Resumes")
        for name, score in accepted:
            st.success(f"{name} - {score}%")

    with col2:
        st.subheader("Rejected Resumes")
        for name, score in rejected:
            st.error(f"{name} - {score}%")
