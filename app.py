from pinecone import Pinecone
import os
import uuid
import google.generativeai as genai
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# Initialize Pinecone with your API key
PINECONE_API_KEY = "pcsk_4KDPSw_HdtwLfPcT7kcYoK7uG6Kz25GJHhNhyrFozXrvraQH7WDy8UYj4q6iCDhb8jnUFc"
GEMINI_API_KEY = "AIzaSyBJT1pauDBLxAP5vcZmV_Ss9we_GpD4TsM"
INDEX_NAME = "resumes"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize Gemini AI
genai.configure(api_key=GEMINI_API_KEY)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to load and chunk resume
def load_resume_chunks(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    chunks = [text[i:i+300] for i in range(0, len(text), 300)]
    return chunks

# Create or connect to the index
if INDEX_NAME in pc.list_indexes().names():
    index = pc.Index(INDEX_NAME)
else:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        metric="cosine"
    )
    index = pc.Index(INDEX_NAME)

# 🔄 Resume Upload
def upload_resume(path):
    chunks = load_resume_chunks(path)
    vectors = []
    for chunk in chunks:
        embedding = embedding_model.encode(chunk).tolist()
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {"text": chunk}
        })
    index.upsert(vectors)
    print(f"✅ Uploaded {len(vectors)} chunks from: {os.path.basename(path)}")

# 💬 Chat About Resume
def chat_about_resume(query):
    query_vector = embedding_model.encode(query).tolist()
    res = index.query(vector=query_vector, top_k=3, include_metadata=True)

    context_chunks = []
    
    for match in res['matches']:
        chunk = match.get("metadata", {}).get("text", "")
        print(f"- {chunk}\n")
        context_chunks.append(chunk)

    prompt = f"""
You are a helpful assistant analyzing resumes.

Here are relevant pieces from the resume:

{chr(10).join(context_chunks)}

Answer the question based on the above resume data:
{query}
"""

    gemini = genai.GenerativeModel("gemini-1.5-pro")
    response = gemini.generate_content(prompt)
    print("🤖 Gemini says:\n")
    print(response.text.strip())

# Main execution block for command line usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Resume Analysis Tool")
    parser.add_argument("--upload", type=str, help="Path to the resume PDF to upload")
    parser.add_argument("--query", type=str, help="Question about the resume")
    
    args = parser.parse_args()
    
    if args.upload:
        upload_resume(args.upload)
    elif args.query:
        chat_about_resume(args.query)
    else:
        print("Please provide either --upload or --query argument")
        print("Example usage:")
        print("  python app.py --upload path/to/resume.pdf")
        print("  python app.py --query \"What skills does this candidate have?\"")