import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import tempfile

model = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="AI Resume Screening", layout="wide")
st.title("AI-Powered Resume Screening System")

job_description = st.text_area("Paste Job Description", height=200)

threshold = st.slider("Shortlisting Threshold (%)", 0, 100, 70)

uploaded_files = st.file_uploader(
    "Upload Resumes (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

def extract_text(file_path):
    text = ""

    if file_path.endswith(".pdf"):
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()

    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + " "

    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    return text


if st.button("Run Screening") and job_description and uploaded_files:

    jd_embedding = model.encode(job_description)

    accepted = []
    rejected = []

    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(uploaded_file.read())
            temp_path = temp.name

        resume_text = extract_text(temp_path)
        resume_embedding = model.encode(resume_text)

        score = cosine_similarity(
            [jd_embedding],
            [resume_embedding]
        )[0][0]

        percentage = round(score * 100, 2)

        if percentage >= threshold:
            accepted.append((uploaded_file.name, percentage))
        else:
            rejected.append((uploaded_file.name, percentage))

    accepted.sort(key=lambda x: x[1], reverse=True)
    rejected.sort(key=lambda x: x[1], reverse=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Accepted Candidates")
        for name, score in accepted:
            st.success(f"{name} — {score}%")

    with col2:
        st.subheader("Rejected Candidates")
        for name, score in rejected:
            st.error(f"{name} — {score}%")

