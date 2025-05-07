import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.embeddings.embeddings import get_embeddings_model  # Assuming this is where your embedding logic resides
from utils.embeddings.vector_store import create_vector_store  # Assuming this is the function for storing embeddings

def render():
    st.header("File Upload Page")
    
    # File uploader allowing PDF, TXT, DOCX files
    uploaded_file = st.file_uploader("Upload your file", type=["pdf", "txt", "docx"])

    if uploaded_file is not None:
        st.success("File uploaded successfully!")

        # Process the file based on its type
        file_extension = uploaded_file.name.split('.')[-1].lower()

        try:
            if file_extension == "pdf":
                # Handle PDF File
                st.info("Processing PDF...")
                # Save the uploaded PDF
                with open("temp_manual.pdf", "wb") as f:
                    f.write(uploaded_file.read())
                # Load and extract text from PDF
                loader = PyPDFLoader("temp_manual.pdf")
                docs = loader.load()
                content_pages = docs[2:]  # Adjust the slice as needed (skip TOC)
                full_text = "\n".join(doc.page_content for doc in content_pages)

            elif file_extension == "txt":
                # Handle TXT File
                st.info("Processing TXT...")
                full_text = uploaded_file.read().decode("utf-8")

            elif file_extension == "docx":
                # Handle DOCX File
                st.info("Processing DOCX...")
                from docx import Document
                doc = Document(uploaded_file)
                full_text = "\n".join([para.text for para in doc.paragraphs])

            else:
                st.error("Unsupported file type.")
                return

            # Split text into chunks for embedding
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.create_documents([full_text])

            if not chunks or all(len(doc.page_content.strip()) == 0 for doc in chunks):
                st.error("No valid content found in the document.")
                return

            # Generate embeddings using the embeddings model
            embeddings_model = get_embeddings_model()

            # Create the vector store with the embeddings
            try:
                create_vector_store(embeddings_model, chunks)
                st.success("Document processed and FAISS index saved.")
            except Exception as e:
                st.error(f"Error creating vector store: {e}")

            # Optionally remove the temp PDF file after processing
            if file_extension == "pdf" and os.path.exists("temp_manual.pdf"):
                os.remove("temp_manual.pdf")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
