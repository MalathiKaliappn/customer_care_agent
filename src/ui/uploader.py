import sys
import os

# Add project root (two levels up from current file) to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.embeddings.embeddings import get_embeddings_model
from utils.embeddings.vector_store import create_vector_store
from utils.file_processor import extract_text_by_extension
import uuid


def render():
    st.header("File Upload Page")

    uploaded_file = st.file_uploader("Upload your file", type=["pdf", "txt", "docx"])

    if uploaded_file is not None:
        st.success("File uploaded successfully!")

        file_extension = uploaded_file.name.split('.')[-1].lower()

        try:
            # Create resources directory if it doesn't exist
            resources_dir = "resources"
            os.makedirs(resources_dir, exist_ok=True)

            # Use UUID to avoid overwriting files with same name
            unique_suffix = uuid.uuid4().hex
            base_name = os.path.splitext(uploaded_file.name)[0]
            save_filename = f"{base_name}_{unique_suffix}.{file_extension}"
            save_path = os.path.join(resources_dir, save_filename)

            # Save uploaded file to the resources directory
            with open(save_path, "wb") as f:
                f.write(uploaded_file.read())

            st.info(f"Processing {file_extension.upper()} file...")

            # Extract text
            full_text = extract_text_by_extension(save_path)

            # Split text into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.create_documents([full_text])

            if not chunks or all(len(doc.page_content.strip()) == 0 for doc in chunks):
                st.error("No valid content found in the document.")
                return

            st.info(f"Document split into {len(chunks)} chunks.")

            # Generate embeddings and create vector store
            embeddings_model = get_embeddings_model()
            create_vector_store(embeddings_model, chunks)

            st.success("Document processed and FAISS index saved.")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
