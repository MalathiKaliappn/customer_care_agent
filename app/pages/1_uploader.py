import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.embeddings import get_embeddings_model
from utils.vector_store import create_vector_store
from langchain.docstore.document import Document

# Streamlit UI
st.set_page_config(page_title="Tesla Manual Assistant", layout="wide")
st.title("📘 Tesla Manual Assistant")
st.markdown("Upload a Tesla Model owner's manual and process it for Q&A.")

uploaded_file = st.file_uploader("Upload the Tesla Manual (PDF)", type="pdf")

if uploaded_file is not None:
    with open("temp_manual.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("Manual uploaded successfully. Processing...")

    # Load and process PDF
    loader = PyPDFLoader("temp_manual.pdf")
    docs = loader.load()
    content_pages = docs[2:]  

    # Combine all text
    full_text = "\n".join(doc.page_content for doc in content_pages)

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([full_text])

    # Generate embeddings
    embeddings_model = get_embeddings_model()  
    create_vector_store(embeddings_model, chunks)

    st.success("Document processed and FAISS index saved.")
else:
    st.info("Please upload a PDF to begin.")
