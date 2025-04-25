# --- MUST BE AT TOP BEFORE ANY OTHER IMPORTS ---
import os
os.environ['STREAMLIT_DISABLE_WATCHER'] = 'true'

# --- Streamlit import and page config ---
import streamlit as st
st.set_page_config(page_title="Resume Analyzer", page_icon="📄", layout="wide")

# --- Other Imports ---
from PyPDF2 import PdfReader
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
import uuid
import logging
from sentence_transformers import SentenceTransformer

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pinecone API Key ---
PINECONE_API_KEY = "pcsk_4KDPSw_HdtwLfPcT7kcYoK7uG6Kz25GJHhNhyrFozXrvraQH7WDy8UYj4q6iCDhb8jnUFc"
INDEX_NAME = "resumes"

# --- Gemini API Key from User ---
st.sidebar.subheader("🔐 Gemini API Key")
user_api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")

if user_api_key:
    try:
        genai.configure(api_key=user_api_key)
        st.sidebar.success("Gemini API key loaded ✅")
    except Exception as e:
        st.sidebar.error(f"Invalid API key: {e}")
else:
    st.sidebar.warning("Please enter your Gemini API key to use the app.")

# --- Initialization ---
pc = Pinecone(api_key=PINECONE_API_KEY)

# Load embedding model
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Embedding model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load embedding model: {e}")
    embedding_model = None

# Initialize Pinecone index
try:
    index = None
    if INDEX_NAME in [i.name for i in pc.list_indexes()]:
        index = pc.Index(INDEX_NAME)
    else:
        if embedding_model:
            dimension = len(embedding_model.encode("test"))
            pc.create_index(
                name=INDEX_NAME,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            index = pc.Index(INDEX_NAME)
except Exception as e:
    logger.error(f"Error initializing Pinecone: {e}")
    st.error("Failed to initialize Pinecone index.")

# --- Helper Functions ---

def extract_text_chunks(file):
    try:
        reader = PdfReader(file)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        return [text[i:i + 500] for i in range(0, len(text), 400)]
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return []

def upload_resume(file, name):
    if not index or not embedding_model:
        st.error("Missing index or embedding model.")
        return 0
    
    chunks = extract_text_chunks(file)
    uploaded = 0

    for i in range(0, len(chunks), 25):
        vectors = []
        for j, chunk in enumerate(chunks[i:i + 25]):
            vector = {
                "id": f"{name}_{i+j}_{uuid.uuid4().hex[:8]}",
                "values": embedding_model.encode(chunk).tolist(),
                "metadata": {
                    "text": chunk,
                    "resume_name": name,
                    "chunk_id": i + j
                }
            }
            vectors.append(vector)
        try:
            index.upsert(vectors=vectors)
            uploaded += len(vectors)
        except Exception as e:
            st.error(f"Error uploading to Pinecone: {e}")
    return uploaded

def query_resume(query, name):
    if not index or not embedding_model:
        return "System not ready."
    
    if not user_api_key:
        return "Gemini API key is missing. Please provide it in the sidebar."

    try:
        genai.configure(api_key=user_api_key)
        query_vec = embedding_model.encode(query).tolist()
        res = index.query(vector=query_vec, top_k=5, include_metadata=True, filter={"resume_name": name})
        chunks = [match.metadata["text"] for match in res.matches if "text" in match.metadata]
        if not chunks:
            return "No relevant data found."
        context = "\n---\n".join(chunks)
        prompt = f"""You are an expert resume analyst. Here's the resume content:

{context}

Answer the question: {query}
Only use information in the resume."""
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error querying model: {e}"

def get_resume_names():
    try:
        res = index.query(vector=[0.0]*384, top_k=1000, include_metadata=True)
        names = list({match.metadata.get("resume_name") for match in res.matches if "resume_name" in match.metadata})
        return sorted(names)
    except Exception as e:
        logger.error(f"Error getting resume names: {e}")
        return []

# --- Streamlit App UI ---
st.title("📄 Resume Analyzer using Pinecone + Gemini AI")
st.markdown("Upload your resume and ask questions about it using AI.")

# --- Upload Section ---
uploaded_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")
manual_name = st.text_input("Enter Resume Name (for new uploads)")

# --- Existing Resumes Dropdown ---
existing_names = get_resume_names()
selected_resume = st.selectbox("Or select an existing uploaded resume:", options=[""] + existing_names)

# Determine final resume name to use
active_resume = manual_name if manual_name else selected_resume

# Upload logic
if uploaded_file and manual_name:
    if "uploaded" not in st.session_state:
        st.session_state.uploaded = False

    if not st.session_state.uploaded:
        with st.spinner("Uploading resume to Pinecone..."):
            total = upload_resume(uploaded_file, manual_name)
            st.success(f"Uploaded {total} chunks for {manual_name}")
            st.session_state.uploaded = True

# --- Query Section ---
if active_resume:
    query = st.text_input("Ask a question about the resume:")
    if query:
        with st.spinner("Thinking..."):
            answer = query_resume(query, active_resume)
            st.subheader("Answer:")
            st.write(answer)
