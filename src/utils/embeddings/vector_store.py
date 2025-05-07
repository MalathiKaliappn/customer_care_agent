import os
import faiss
import streamlit as st
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.docstore.document import Document

def create_vector_store(embeddings, chunks, index_path="faiss_index"):
    """
    Creates and saves a FAISS vector store from the provided document chunks.
    """
    # Initialize FAISS index through LangChain
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)

    # Save vector store to disk
    os.makedirs(index_path, exist_ok=True)

    try:
        vector_store.save_local(index_path)
        print(f"✅ Vector store saved to {index_path}")
    except Exception as e:
        print(f"Error saving vector store: {e}")
        st.error(f"Failed to save vector store.\nDetails: {e}")

    return vector_store



def load_vector_store(embeddings, index_path="faiss_index"):
    """
    Loads an existing FAISS vector store from disk.
    Raises a FileNotFoundError if the index is missing.
    """
    # Check if the FAISS index file exists
    index_file = os.path.join(index_path, "index.faiss")
    if not os.path.exists(index_file):
        raise FileNotFoundError(
            f"FAISS index not found at '{index_file}'.\n"
            "➡️ Please run the uploader page to generate the index first."
        )

    # Load the FAISS vector store
    try:
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading vector store: {e}")
        raise
