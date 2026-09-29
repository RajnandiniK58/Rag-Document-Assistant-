import streamlit as st
from pypdf import PdfReader
import docx
import pandas as pd
import io
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from chunking.strategies import fixed_size_chunk, sentence_chunk, paragraph_chunk, semantic_chunk
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def generate_answer(query, retrieved_chunks):
    """Send retrieved chunks + query to Gemini and get a grounded answer."""
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""You are a helpful assistant answering questions based ONLY on the provided context.
If the answer isn't in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""
    model = genai.GenerativeModel("gemini-2.5-flash")
    # model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def build_faiss_index(chunks, model):
    """Embed all chunks and build a FAISS index for similarity search."""
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')

    dimension = embeddings.shape[1]  # 384 for MiniLM-L6-v2
    index = faiss.IndexFlatL2(dimension)  # L2 = Euclidean distance search
    index.add(embeddings)

    return index, embeddings

def retrieve_top_chunks(query, chunks, index, model, k=8):
    """Given a query, find the k most similar chunks."""
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k)

    results = [chunks[i] for i in indices[0]]
    return results, distances[0]
def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def load_docx(file):
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def load_txt(file):
    return file.read().decode("utf-8")

def load_csv(file):
    df = pd.read_csv(file)
    # Convert each row into a readable text line
    text = df.to_string(index=False)
    return text

def load_document(uploaded_file):
    file_type = uploaded_file.name.split(".")[-1].lower()
    if file_type == "pdf":
        return load_pdf(uploaded_file)
    elif file_type == "docx":
        return load_docx(uploaded_file)
    elif file_type == "txt":
        return load_txt(uploaded_file)
    elif file_type == "csv":
        return load_csv(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

st.title("RAG-Powered Document Intelligence Assistant")

model = load_embedding_model()

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt", "csv"])
chunking_method = st.selectbox("Chunking Strategy", ["Fixed-size", "Sentence", "Paragraph", "Semantic"])

if uploaded_file:
    text = load_document(uploaded_file)
    st.success(f"Extracted {len(text)} characters")

    if chunking_method == "Fixed-size":
        chunks = fixed_size_chunk(text)
    elif chunking_method == "Sentence":
        chunks = sentence_chunk(text)
    elif chunking_method == "Paragraph":
        chunks = paragraph_chunk(text)
    else:
        chunks = semantic_chunk(text, model)

    st.info(f"Created {len(chunks)} chunks using {chunking_method} strategy")

    index, embeddings = build_faiss_index(chunks, model)
    st.session_state["chunks"] = chunks
    st.session_state["index"] = index
    st.session_state["model"] = model

    query = st.text_input("Ask a question about the document")
    if query:
        results, distances = retrieve_top_chunks(query, chunks, index, model)

        with st.spinner("Generating answer..."):
            answer = generate_answer(query, results)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("View source chunks"):
            for i, (chunk, dist) in enumerate(zip(results, distances)):
                st.markdown(f"**Source {i+1}** (distance: {dist:.4f})")
                st.text(chunk[:300])