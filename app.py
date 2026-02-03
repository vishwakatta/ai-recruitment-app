import streamlit as st
import os
import tempfile
import re
import smtplib
from email.message import EmailMessage

import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq


# ==============================
# PAGE + UI STYLE
# ==============================
st.set_page_config(page_title="AI Resume Screening", layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
.card {
    background-color: #111827;
    padding: 16px;
    border-radius: 10px;
    margin-bottom: 12px;
}
.accepted {
    border-left: 6px solid #22c55e;
}
.rejected {
    border-left: 6px solid #ef4444;
}
</style>
""", unsafe_allow_html=True)


# ==============================
# LOAD ML MODEL
# ==============================
model = SentenceTransformer("all-MiniLM-L6-v2")


# ==============================
# HELPER FUNCTIONS
# ==============================
def extract_text(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def extract_email(text):
    match = re.search(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text
    )
    return match.group(0) if match else None


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
            "Our HR team will contact you with next steps.\n\n"
            "Regards,\nHR Team"
        )
    else:
        msg["Subject"] = "Application Status Update"
        msg.set_content(
            "Dear Candidate,\n\n"
            "Thank you for your interest. After careful review, "
            "we will not be moving forward at this stage.\n\n"
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

Explain the decision in bullet points (2–4 points).
Be clear, factual, and professional.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content


# ==============================
# UI – INPUT SECTION
# ==============================
st.title("🤖 AI Resume Screening Platform")
st.caption("AI-powered shortlisting with explainable decisions")

st.subheader("📄 Job Details")

job_description = st.text_area(
    "Job Description",
    height=180,
    placeholder="Paste the job description here..."
)

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_files = st.file_uploader(
        "Upload Resumes (PDF only)",
        type=["pdf"],
        accept_multiple_files=True
    )

with col2:
    threshold = st.slider("Shortlisting Threshold (%)", 0, 100, 70)
    run = st.button("🚀 Run Screening", use_container_width=True)


# ==============================
# MAIN LOGIC
# ==============================
if run and job_description and uploaded_files:

    jd_embedding = model.encode(job_description)

    accepted = []
    rejected = []

    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(uploaded_file.getvalue())
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
        else:
            status = "Rejected"

        explanation = generate_ai_explanation(
            job_description,
            resume_text,
            percentage,
            status
        )

        email = extract_email(resume_text)
        if email:
            try:
                send_status_email(email, status)
                st.success(f"Email sent to {email} ({status})")
            except Exception as e:
                st.error(f"Email failed for {email}: {e}")
        else:
            st.warning(f"No email found in {uploaded_file.name}")

        if status == "Selected":
            accepted.append((uploaded_file.name, percentage, explanation))
        else:
            rejected.append((uploaded_file.name, percentage, explanation))


    # ==============================
    # RESULTS SECTION
    # ==============================
    st.divider()
    colA, colB = st.columns(2)

    with colA:
        st.subheader("✅ Accepted Candidates")
        for name, score, explanation in accepted:
            st.markdown(f"""
            <div class="card accepted">
                <h4>{name}</h4>
                <b>Match Score:</b> {score}%
            </div>
            """, unsafe_allow_html=True)

            for line in explanation.split("\n"):
                if line.strip():
                    st.markdown(f"- {line.strip()}")

    with colB:
        st.subheader("❌ Rejected Candidates")
        for name, score, explanation in rejected:
            st.markdown(f"""
            <div class="card rejected">
                <h4>{name}</h4>
                <b>Match Score:</b> {score}%
            </div>
            """, unsafe_allow_html=True)

            for line in explanation.split("\n"):
                if line.strip():
                    st.markdown(f"- {line.strip()}")
