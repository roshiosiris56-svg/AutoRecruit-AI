import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PyPDF2 import PdfReader

# -----------------------------
# CONFIGURATION
# -----------------------------
RESUME_FOLDER = "resumes"
DEFAULT_TOP_K = 10

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="AutoRecruit AI",
    layout="wide"
)

st.title("AutoRecruit AI")
st.subheader("Semantic Resume Search for Hiring Teams")

# -----------------------------
# MODEL LOADING
# -----------------------------
@st.cache_resource

def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -----------------------------
# CREATE RESUME FOLDER IF MISSING
# -----------------------------
os.makedirs(RESUME_FOLDER, exist_ok=True)

# -----------------------------
# FILE UPLOAD SECTION
# -----------------------------
st.header("1. Upload Candidate Resumes")

uploaded_files = st.file_uploader(
    "Upload PDF resumes",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        save_path = os.path.join(RESUME_FOLDER, uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.success(f"{len(uploaded_files)} resume(s) uploaded successfully.")

# -----------------------------
# READ PDF TEXT
# -----------------------------
def extract_text_from_pdf(pdf_path):
    text = ""

    try:
        reader = PdfReader(pdf_path)

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "

    except Exception as e:
        st.error(f"Error reading {pdf_path}: {e}")

    return text

# -----------------------------
# LOAD RESUMES
# -----------------------------
resume_files = [
    f for f in os.listdir(RESUME_FOLDER)
    if f.endswith('.pdf')
]

resume_texts = []
resume_names = []

for file_name in resume_files:
    path = os.path.join(RESUME_FOLDER, file_name)

    text = extract_text_from_pdf(path)

    if text.strip():
        resume_texts.append(text)
        resume_names.append(file_name)

# -----------------------------
# EMBEDDINGS + FAISS
# -----------------------------
index = None
resume_embeddings = None

if resume_texts:
    resume_embeddings = model.encode(resume_texts)

    embedding_dimension = resume_embeddings.shape[1]

    index = faiss.IndexFlatL2(embedding_dimension)
    index.add(np.array(resume_embeddings).astype('float32'))

# -----------------------------
# SEMANTIC SEARCH
# -----------------------------
st.header("2. AI Candidate Search")

query = st.text_input(
    "Enter hiring requirement",
    placeholder="Example: Need 10 Full Stack Developers in Bangalore with React and Node.js experience"
)

if st.button("Search Candidates"):

    if not resume_texts:
        st.warning("Please upload resumes first.")

    elif not query.strip():
        st.warning("Please enter a hiring requirement.")

    else:
        import re

        match = re.search(r'\b(\d+)\b', query)

        if match:
            requested_count = int(match.group(1))
        else:
            requested_count = DEFAULT_TOP_K

        top_k = min(requested_count, len(resume_texts))

        query_embedding = model.encode([query])

        distances, indices = index.search(
            np.array(query_embedding).astype('float32'),
            top_k
        )

        st.header("Top Matching Candidates")

        for rank, idx in enumerate(indices[0]):
            candidate_name = resume_names[idx]

            similarity_score = float(1 / (1 + distances[0][rank]))
            similarity_percentage = round(similarity_score * 100, 2)

            with st.container(border=True):
                st.subheader(f"#{rank + 1} - {candidate_name}")

                st.write(f"Match Score: {similarity_percentage}%")

                st.progress(min(similarity_score, 1.0))

# -----------------------------
# SIDEBAR INFO
# -----------------------------
st.sidebar.title("AutoRecruit AI")

st.sidebar.info(
    "This Proof-of-Concept demonstrates AI-powered semantic resume search using embeddings and vector similarity."
)

st.sidebar.markdown("### Features")

st.sidebar.markdown(
    """
- Resume Upload
- PDF Parsing
- Semantic Search
- FAISS Vector Database
- AI Candidate Ranking
- Similarity Scoring
"""
)
