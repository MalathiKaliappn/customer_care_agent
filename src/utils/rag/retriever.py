# def get_retriever(vector_store, k=3):
#     return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})


def get_retriever(vector_store, k=3, search_type="similarity"):
    """
    Creates a retriever from the provided vector store to fetch similar documents.
    
    Args:
        vector_store: The FAISS or other vector store instance containing document embeddings.
        k (int): The number of similar documents to retrieve (default: 3).
        search_type (str): The search type used for similarity (default: "similarity").

    Returns:
        The retriever for querying the vector store.
    """
    if not hasattr(vector_store, "as_retriever"):
        raise ValueError("The provided vector_store does not have an 'as_retriever' method.")

    # Create and return the retriever
    return vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k})
