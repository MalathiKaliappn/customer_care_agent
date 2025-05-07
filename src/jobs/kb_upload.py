import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from customer_care_agent.src.utils.embeddings.embeddings import get_embeddings_model
from customer_care_agent.src.utils.embeddings.vector_store import create_vector_store

# Use relative path to the manual PDF
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Adjust if needed
PDF_PATH = BASE_DIR / "resources" / "tesla-owner-manual.pdf"

def main():
    # Check if the PDF file exists at the relative path
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found at: {PDF_PATH}")

    print(f"Loading PDF from: {PDF_PATH}")
    loader = PyPDFLoader(str(PDF_PATH))
    docs = loader.load()
    content_pages = docs[2:]  # Skip TOC if desired

    print("Splitting document into chunks...")
    text = "\n".join(doc.page_content for doc in content_pages)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([text])

    print("Creating embeddings...")
    embeddings_model = get_embeddings_model()

    print("Creating and saving FAISS vector store...")
    create_vector_store(embeddings_model, chunks)

    print("Upload and indexing complete.")

if __name__ == "__main__":
    main()
