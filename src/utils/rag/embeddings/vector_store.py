# import os
# import faiss
# from langchain_community.vectorstores import FAISS
# from langchain_community.docstore.in_memory import InMemoryDocstore

# def create_vector_store(embeddings, chunks, index_path="faiss_index"):
#     """
#     Creates and saves a FAISS vector store from the provided document chunks.
#     """
    
#     sample_embedding = embeddings.embed_query("test")
#     embedding_dim = len(sample_embedding)

#     # Create FAISS index
#     index = faiss.IndexFlatL2(embedding_dim)

#     # Initialize the vector store
#     vector_store = FAISS(
#         embedding_function=embeddings,
#         index=index,
#         docstore=InMemoryDocstore(),
#         index_to_docstore_id={},
#     )

    
#     vector_store.add_documents(chunks)

#     # Save the index
#     vector_store.save_local(index_path)
#     return vector_store

# def load_vector_store(embeddings, index_path="faiss_index"):
#     """
#     Loads an existing FAISS vector store from disk.
#     Raises a FileNotFoundError with explanation if the index is missing.
#     """
#     index_file = os.path.join(index_path, "index.faiss")
#     if not os.path.exists(index_file):
#         raise FileNotFoundError(
#             f"FAISS index not found at '{index_file}'.\n"
#             "➡️ Please run the uploader page to generate the index first."
#         )

#     return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

import os
import faiss
from langchain.embeddings import HuggingFaceEmbeddings  # Ensure this import is correct
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.docstore.document import Document

def create_vector_store(embeddings, chunks, index_path="faiss_index"):
    """
    Creates and saves a FAISS vector store from the provided document chunks.
    """

    # Get the dimensionality of the embeddings
    sample_embedding = embeddings.embed_query("test")
    embedding_dim = len(sample_embedding)

    # Create FAISS index for similarity search
    index = faiss.IndexFlatL2(embedding_dim)

    # Initialize the FAISS vector store
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    # Convert raw chunks to Documents (if chunks are not already Documents)
    # If `chunks` are raw text strings, wrap them in Document objects
    #documents = [Document(page_content=chunk) for chunk in chunks]

    # Add the documents to the vector store
    vector_store.add_documents(chunks)

    # Save the vector store locally
    if not os.path.exists(index_path):
        os.makedirs(index_path)
    vector_store.save_local(index_path)

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
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
