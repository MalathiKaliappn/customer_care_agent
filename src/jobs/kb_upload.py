import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.rag.embeddings.embeddings import get_embeddings_model
from utils.rag.embeddings.vector_store import create_vector_store

# Path to the manual PDF
PDF_PATH = "C:\Users\malat\Desktop\ML\Tesla_RAG/tesla-owner-manual.pdf"

def main():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found at: {PDF_PATH}")

    print(f"Loading PDF from: {PDF_PATH}")
    loader = PyPDFLoader(PDF_PATH)
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
